"""Graph transformer on all seven maps: the train_thgnn_allmaps.py pipeline with the GAT encoder replaced by a graph transformer.

DRAFT, NOT RUN. Pre-registration is not done yet. Before any run whose result will be read, the decision rule and the step budget
must be written into a pin, and the pin must exist before the first result is opened. Until then, nothing this script writes may be
read or used to make a choice. The candidate rule is the one in train_thgnn_allmaps.py (mean paired log-loss difference on the
de_mirage test seconds <= -0.0005 and 95% match-cluster CI upper bound < 0). It is a candidate only, not an agreed rule.
Pin id: not yet written.

Run on the pod (the repo is at /workspace/csds, so the data is under /workspace/csds/data/round_state and data/combined):
  python -I -X utf8 scripts/model/train_graph_transformer.py --seed 0 --graph kmeans --out_dir /workspace/out/gt_kmeans_s0

Same as train_thgnn_allmaps.py:
  - data: the seven maps, round-state files <map>_ext.npz, loadout <map>_ext_loadout.npy, snapshot cache
    cache_snapshot_2026_ext.parquet; 20-second window at 1 Hz; 21 features per slot (6 state, 12 loadout, 3 velocity); 12
    round-state inputs (3 state, 2 alive fractions, 7 map one-hot); float16 device storage, float32 arithmetic.
  - split: training uses the train split of every map; validation uses the validation split of every map (best checkpoint); the
    test is the frozen de_mirage test seconds, read once after training.
  - loss and optimiser: masked BCE over usable steps, Adam at 1e-3, batch 512; the current second is the last window step.
  - metrics: log loss clipped at 1e-6 (as compare_paired_de_mirage.py); 95% intervals from a match-cluster bootstrap (wp_common.N_BOOT resamples).

Different on purpose:
  - Graph encoder: ten player tokens and one round-state token go through three pre-norm transformer blocks (multi-head
    self-attention, four heads). The map graph enters as a learned attention bias, not a mask. For each pair of players, the hop
    distance between their assigned map nodes along the adjacency (capped at 6, plus one bucket for unreachable pairs) indexes a
    learned per-head bias in every layer. Round-token pairs have three more buckets. Dead players are masked as keys only.
  - Each player token also gets a learned embedding of the map node it is assigned to, per map (the map_base of the GAT model).

Outputs in --out_dir, with tag allmaps_gt_<graph>_seed<seed>:
  <tag>_metrics.json       validation and test log loss with intervals, history, selected step, budget
  <tag>_pred.npz           de_mirage rows in the train_thgnn_allmaps.py layout (scored by compare_paired_de_mirage.py)
  <tag>_test_preds.parquet match_key, round, second, label, p for the de_mirage test seconds (train_temporal_allmaps.py columns)
  <tag>.pt                 best checkpoint

--steps sets the total optimizer steps (default 50,000, the GAT script's budget, about one pass at batch 512). It overrides
epochs x rows-per-epoch. --steps 0 falls back to --epochs, which counts passes over the all-maps training split. The script
prints the training row count and the step count before training starts. The pin must fix the step count, not only the epoch count.
--graph navgraph needs <map>_ext_navgraph.npz for all seven maps. Only de_mirage has one now, so the script stops before loading.
"""
import argparse
import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_temporal_round_model as ttr  # noqa: E402  (build_table, previous_rows, split constants, DATA and OUT)
import train_temporal_allmaps as T  # noqa: E402  (MAPS, CACHE)
import train_thgnn_allmaps as TA  # noqa: E402  (window, feature and node constants shared with the GAT model)
import wp_common as W  # noqa: E402  (EPS, N_BOOT, ll_vec: the shared log-loss rule)

MAPS = T.MAPS
CACHE = T.CACHE
DATA, OUT = ttr.DATA, ttr.OUT
TRAIN, VAL, TEST = ttr.TRAIN, ttr.VAL, ttr.TEST
FROZEN_KEYS = os.environ.get("FROZEN_KEYS", "/mnt/g/pureskill/csds-cs2/data/benchmark/frozen_test_keys.json")
LOOKBACK, N_SLOTS, FP, G_DIM, CHUNK = TA.LOOKBACK, TA.N_SLOTS, TA.FP, TA.G_DIM, TA.CHUNK
N_MAPS = len(MAPS)
EPS, N_BOOT = 1e-6, W.N_BOOT  # clip as in compare_paired_de_mirage.py:26 and train_thgnn_allmaps.py:273 (wp_common.EPS is 1e-15)
D_MODEL, HEADS, LAYERS, FFN, LATENT, GRU_H = 64, 4, 3, 128, 64, 64
MAX_HOP = 6  # hop distances above this share one bucket
N_HOP = MAX_HOP + 2  # buckets 0..MAX_HOP for reachable pairs, then one bucket for unreachable pairs
N_BUCKETS = N_HOP + 3  # plus player-to-round, round-to-player and round-to-round
N_TOK = N_SLOTS + 1  # ten players, then the round-state token
GRAPH_SUFFIX = {"kmeans": "_ext_mapgraph.npz", "navgraph": "_ext_navgraph.npz"}


def ll_vec(p, y):
    """Per-row log loss clipped at EPS (1e-6). Same form as wp_common.ll_vec, which clips at 1e-15 and is shared with other scripts."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def hop_buckets(A):
    """Hop distance between map nodes along the adjacency (A[i, j] is an edge i -> j), capped at MAX_HOP; unreachable -> N_HOP - 1."""
    K = A.shape[0]
    D = np.where(A, 1.0, np.inf)
    np.fill_diagonal(D, 0.0)
    for k in range(K):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return np.where(np.isfinite(D), np.minimum(D, MAX_HOP), N_HOP - 1).astype(np.int64)


class GTBlock(nn.Module):
    """Pre-norm transformer block. The attention logits get a learned bias per (bucket, head), indexed by the pair's bucket."""

    def __init__(self, d=D_MODEL, heads=HEADS, ffn=FFN):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.bias = nn.Embedding(N_BUCKETS, heads)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x, bucket, key_dead):
        # x (G, N, d); bucket (G, N, N) long, query by key; key_dead (G, N) bool, True for a player who is not alive
        G_, N, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(G_, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)  # (G, H, N, N)
        logits = logits + self.bias(bucket).permute(0, 3, 1, 2)
        logits = logits.masked_fill(key_dead[:, None, None, :], -1e9)
        att = torch.softmax(logits, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(G_, N, d)
        x = x + self.proj(o)
        return x + self.ff(self.ln2(x))


class GraphTransformer(nn.Module):
    def __init__(self, K, fp=FP, n_maps=N_MAPS, gfeat=G_DIM):
        super().__init__()
        self.K = K
        self.player_enc = nn.Sequential(nn.Linear(fp, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, D_MODEL))
        self.slot = nn.Parameter(torch.randn(N_SLOTS, D_MODEL) * 0.02)  # which of the ten slots (five CT, five T)
        self.node_base = nn.Parameter(torch.zeros(n_maps, K, D_MODEL))  # per map, the map node a player is assigned to
        self.round_enc = nn.Linear(gfeat, D_MODEL)
        self.round_tok = nn.Parameter(torch.zeros(D_MODEL))
        self.blocks = nn.ModuleList([GTBlock() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.readout = nn.Sequential(nn.Linear(3 * D_MODEL + gfeat, LATENT), nn.ReLU())
        self.gru = nn.GRU(LATENT, GRU_H, batch_first=True)
        self.head = nn.Sequential(nn.Linear(GRU_H, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, Pw, alive_w, assign_w, G_w, mapb, hop_all, step_mask):
        # Pw (B, T, 10, fp); alive_w (B, T, 10) float; assign_w (B, T, 10) long, map node ids; G_w (B, T, gfeat);
        # mapb (B,) long, map id of each window; hop_all (n_maps, K*K) long buckets; step_mask (B, T) float
        B, T_ = alive_w.shape[:2]
        dev = alive_w.device
        a = alive_w.unsqueeze(-1)
        hp = (self.player_enc(Pw) + self.slot + self.node_base[mapb[:, None, None], assign_w]) * a  # (B, T, 10, d)
        r = (self.round_enc(G_w) + self.round_tok).unsqueeze(2)  # (B, T, 1, d)
        x = torch.cat([hp, r], dim=2).reshape(B * T_, N_TOK, D_MODEL)
        # bucket of each (query, key) pair: player pairs from the map graph; round-token pairs get their own buckets
        flat = assign_w.unsqueeze(-1) * self.K + assign_w.unsqueeze(-2)  # (B, T, 10, 10), index into K*K
        pp = hop_all[mapb[:, None, None, None], flat]  # (B, T, 10, 10)
        to_round = torch.full((B, T_, N_SLOTS, 1), N_HOP, dtype=torch.long, device=dev)
        from_round = torch.full((B, T_, 1, N_SLOTS), N_HOP + 1, dtype=torch.long, device=dev)
        round_round = torch.full((B, T_, 1, 1), N_HOP + 2, dtype=torch.long, device=dev)
        bucket = torch.cat([torch.cat([pp, to_round], dim=3),
                            torch.cat([from_round, round_round], dim=3)], dim=2).reshape(B * T_, N_TOK, N_TOK)
        key_dead = torch.cat([alive_w < 0.5, torch.zeros(B, T_, 1, dtype=torch.bool, device=dev)], dim=2).reshape(B * T_, N_TOK)
        for blk in self.blocks:
            x = blk(x, bucket, key_dead)
        x = self.ln_f(x).reshape(B, T_, N_TOK, D_MODEL)
        ct = (x[:, :, :5] * a[:, :, :5]).sum(dim=2)
        tt = (x[:, :, 5:N_SLOTS] * a[:, :, 5:]).sum(dim=2)
        rt = x[:, :, N_SLOTS]
        z = self.readout(torch.cat([ct, tt, rt, G_w], dim=-1)) * step_mask.unsqueeze(-1)
        out, _ = self.gru(z)
        return self.head(out).squeeze(-1)  # (B, T) logits, oldest first


def load(device, graph):
    """Every map's seconds in one set of device arrays. The same construction as train_thgnn_allmaps.load, with the graph file
    chosen by --graph and the hop table built from each map's adjacency. Rows are in MAPS order, so de_mirage comes first."""
    suffix = GRAPH_SUFFIX[graph]
    lens = [np.load(DATA / f"{m}_ext_loadout.npy", mmap_mode="r").shape[0] for m in MAPS]
    N = int(sum(lens))
    feat = torch.empty((N, N_SLOTS, FP), dtype=torch.float16, device=device)
    G = torch.zeros((N, G_DIM), dtype=torch.float16, device=device)
    assign = torch.zeros((N, N_SLOTS), dtype=torch.int8, device=device)
    label = np.zeros(N, dtype=np.float32)
    split = np.zeros(N, dtype=np.int8)
    usable = np.zeros(N, dtype=bool)
    map_id = np.zeros(N, dtype=np.int64)
    round_id = np.zeros(N, dtype=np.int64)
    second = np.zeros(N, dtype=np.int64)
    match_idx = np.zeros(N, dtype=np.int64)
    prev = np.empty((N, LOOKBACK), dtype=np.int32)
    hops, keys = [], []
    K = None
    key_offset, s = 0, 0
    for mi, m in enumerate(MAPS):
        arr, X, table, lab, st, use = ttr.build_table(m, f"{m}_ext", CACHE)
        n = X.shape[0]
        lo = np.load(DATA / f"{m}_ext_loadout.npy", mmap_mode="r")
        mg = np.load(DATA / f"{m}{suffix}")
        A_m = mg["adjacency"].astype(bool)
        if K is None:
            K = A_m.shape[0]
            if K > 127:
                raise SystemExit(f"{K} map nodes do not fit the int8 assignment store")
        if A_m.shape != (K, K):
            raise SystemExit(f"{m}: graph has {A_m.shape[0]} nodes, the first map has {K}; all maps must share one node count")
        hops.append(hop_buckets(A_m).reshape(-1))
        am = mg["assign"].astype(np.int64)
        assert am.shape == (n, N_SLOTS), (m, am.shape, n)
        loc = ttr.previous_rows(arr["match_idx"], arr["round"], arr["second"], LOOKBACK)  # map-local row indices
        prev1 = loc[:, 1]
        has_prev = prev1 >= 0
        idx_prev = np.where(has_prev, prev1, 0)
        onehot = np.zeros((1, N_MAPS), dtype=np.float32)
        onehot[0, mi] = 1.0
        for c in range(0, n, CHUNK):
            e = min(n, c + CHUNK)
            Xc = np.asarray(X[c:e], dtype=np.float32)
            ip = idx_prev[c:e]
            both = has_prev[c:e, None] & (Xc[:, :, 0] > 0.5) & (X[ip, :, 0] > 0.5)
            dv = Xc[:, :, 1:4] - X[ip, :, 1:4]
            vel = np.where(both[..., None], dv, 0.0).astype(np.float32)
            chunk = np.concatenate([Xc, np.asarray(lo[c:e], dtype=np.float32), vel], axis=2)
            feat[s + c:s + e] = torch.as_tensor(chunk, device=device).half()
            gc = np.concatenate([np.asarray(st[c:e], dtype=np.float32),
                                 Xc[:, :5, 0].mean(axis=1, keepdims=True), Xc[:, 5:, 0].mean(axis=1, keepdims=True),
                                 np.repeat(onehot, e - c, axis=0)], axis=1)
            G[s + c:s + e] = torch.as_tensor(gc, device=device).half()
            assign[s + c:s + e] = torch.as_tensor(np.maximum(am[c:e], 0).astype(np.int8), device=device)
        label[s:s + n] = lab
        split[s:s + n] = arr["split"].astype(np.int8)
        usable[s:s + n] = use
        map_id[s:s + n] = mi
        round_id[s:s + n] = arr["round"]
        second[s:s + n] = arr["second"]
        match_idx[s:s + n] = arr["match_idx"] + key_offset
        prev[s:s + n] = np.where(loc >= 0, loc + s, -1).astype(np.int32)
        keys.append(arr["match_keys"])
        key_offset += arr["match_keys"].size
        print(f"  {m}: {n:,} seconds, {arr['match_keys'].size:,} matches, usable {int(use.sum()):,} "
              f"(device {torch.cuda.memory_allocated() / 1e9:.1f} GB)", flush=True)
        s += n
        del X, lo, arr, table, st, loc, am, mg
    return {"feat": feat, "G": G, "assign": assign, "hop_all": torch.as_tensor(np.stack(hops), dtype=torch.long, device=device),
            "K": K, "label_t": torch.as_tensor(label, device=device), "usable_t": torch.as_tensor(usable, device=device),
            "map_t": torch.as_tensor(map_id, device=device),
            "label": label, "split": split, "usable": usable, "map_id": map_id, "round": round_id, "second": second,
            "match_idx": match_idx, "prev": prev, "keys": np.concatenate(keys), "N": N, "n0": int(lens[0])}


def match_ci(loss_rows, match_ids, seed):
    """95% interval of the row-weighted mean loss, resampling matches (the unit of dependence), N_BOOT resamples, seeded."""
    uniq, inv = np.unique(match_ids, return_inverse=True)
    M = uniq.size
    sums = np.bincount(inv, weights=loss_rows, minlength=M)
    counts = np.bincount(inv, minlength=M).astype(np.float64)
    idx = np.random.default_rng(seed).integers(0, M, size=(N_BOOT, M))
    reps = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], int(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph", choices=sorted(GRAPH_SUFFIX), default="kmeans", help="map graph: k-means nodes or navigation-mesh regions")
    ap.add_argument("--out_dir", default=str(OUT / "graph_transformer"))
    ap.add_argument("--steps", type=int, default=50000, help="total optimizer steps; overrides epochs x rows-per-epoch (0 uses --epochs)")
    ap.add_argument("--epochs", type=int, default=1, help="passes over the all-maps training split, used only with --steps 0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=10000, help="validate every this many steps, and at the end")
    ap.add_argument("--test", action="store_true", help="also read the de_mirage test split once after training (default off: validation only)")
    ap.add_argument("--test_frozen", action="store_true", help="read the frozen test matches on all seven maps once after training (implies --test)")
    a = ap.parse_args()
    if a.test_frozen:
        a.test = True
    if a.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if a.steps < 0:
        raise SystemExit("--steps must be 0 or more")
    suffix = GRAPH_SUFFIX[a.graph]
    absent = [f"{m}{suffix}" for m in MAPS if not (DATA / f"{m}{suffix}").exists()]
    if absent:
        raise SystemExit(f"graph files missing in {DATA}: {', '.join(absent)}")
    tag = f"allmaps_gt_{a.graph}_seed{a.seed}"
    out_dir = Path(a.out_dir)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device(a.device)
    print(f"device {device}; torch {torch.__version__}; seed {a.seed}; graph {a.graph}; epochs {a.epochs}; batch {a.batch}",
          flush=True)
    t0 = time.time()
    D = load(device, a.graph)
    hop_all, K = D["hop_all"], D["K"]
    print(f"rows {D['N']:,} loaded in {time.time() - t0:.0f} s; device {torch.cuda.memory_allocated() / 1e9:.1f} GB; "
          f"map nodes {K}", flush=True)

    train_rows = np.flatnonzero((D["split"] == TRAIN) & D["usable"])
    val_rows = np.flatnonzero((D["split"] == VAL) & D["usable"])
    test_rows = np.flatnonzero((D["split"] == TEST) & D["usable"] & (D["map_id"] == 0))  # de_mirage test only
    if a.test_frozen:  # the frozen test matches on all seven maps; the split's other test matches are left out
        frozen = set(json.load(open(FROZEN_KEYS)))
        in_frozen = np.array([k in frozen for k in D["keys"]], dtype=bool)
        test_rows = np.flatnonzero((D["split"] == TEST) & D["usable"] & in_frozen[D["match_idx"]])
    label_val = D["label"][val_rows]
    steps = a.steps if a.steps > 0 else a.epochs * -(-train_rows.size // a.batch)
    print(f"training rows {train_rows.size:,}; steps {steps:,} ({'--steps' if a.steps > 0 else '--epochs'}); "
          f"val {val_rows.size:,} (all maps), test {test_rows.size:,} (de_mirage)", flush=True)

    def gather(rows):
        idx = D["prev"][rows]  # (B, T); k = 0 is the row itself, larger k further into the past
        mk_np = idx >= 0
        safe = np.where(mk_np, idx, 0)[:, ::-1].copy()  # oldest step first, as in train_thgnn_allmaps.py
        mk_np = mk_np[:, ::-1].copy()
        ix = torch.as_tensor(safe, dtype=torch.long, device=device)
        mk = torch.as_tensor(mk_np, dtype=torch.float32, device=device)
        Pw = D["feat"][ix].float() * mk[..., None, None]
        alive_w = (Pw[..., 0] > 0.5).float()
        assign_w = D["assign"][ix].long()
        G_w = D["G"][ix].float() * mk[..., None]
        used = D["usable_t"][ix] & (mk > 0.5)
        y = D["label_t"][ix]
        mapb = D["map_t"][torch.as_tensor(rows, dtype=torch.long, device=device)]  # one match per window, hence one map
        return Pw, alive_w, assign_w, G_w, mapb, mk, used, y

    model = GraphTransformer(K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    @torch.no_grad()
    def predict(rows, batch=2048):
        model.eval()
        out = np.empty(rows.size, dtype=np.float64)
        for s in range(0, rows.size, batch):
            r = rows[s:s + batch]
            Pw, alive_w, assign_w, G_w, mapb, mk, _, _ = gather(r)
            logit = model(Pw, alive_w, assign_w, G_w, mapb, hop_all, mk)[:, -1]  # the current second is the newest step
            out[s:s + batch] = torch.sigmoid(logit.double()).cpu().numpy()
        model.train()
        return out

    history = []
    best_val, best_step, best_state, best_p_val = float("inf"), 0, None, None
    order, pos = rng.permutation(train_rows), 0
    running, count = 0.0, 0.0
    step = 0
    t_train = time.time()

    def validate():
        nonlocal best_val, best_step, best_state, best_p_val
        p_val = predict(val_rows)
        v = float(ll_vec(p_val, label_val).mean())
        history.append({"step": step, "val_log_loss": v, "train_loss": running / max(count, 1.0),
                        "seconds": round(time.time() - t_train, 1)})
        print(json.dumps(history[-1]), flush=True)
        if v < best_val:
            best_val, best_step, best_p_val = v, step, p_val
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    while step < steps:
        if pos >= order.size:
            order, pos = rng.permutation(train_rows), 0
        r = order[pos:pos + a.batch]
        pos += a.batch
        Pw, alive_w, assign_w, G_w, mapb, mk, used, y = gather(r)
        logit = model(Pw, alive_w, assign_w, G_w, mapb, hop_all, mk)
        per_step = bce(logit, y) * used.float()
        loss = per_step.sum() / used.float().sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += float(per_step.sum().item())
        count += float(used.float().sum().item())
        step += 1
        if step % 500 == 0:
            print(f"  step {step}/{steps}, train loss {running / max(count, 1.0):.4f}, {time.time() - t_train:.0f} s "
                  f"(device {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak)", flush=True)
        if step % a.eval_every == 0 or step == steps:
            validate()
            running, count = 0.0, 0.0

    # selection used validation only; the test split is read only with --test (off by default: validation-only run)
    loss_val = ll_vec(best_p_val, label_val)
    val_ci, val_matches = match_ci(loss_val, D["match_idx"][val_rows], a.seed)
    test_ll = test_ci = test_matches = None
    if a.test:
        model.load_state_dict(best_state)
        p_test = predict(test_rows)
        loss_test = ll_vec(p_test, D["label"][test_rows])
        test_ll = float(loss_test.mean())
        test_ci, test_matches = match_ci(loss_test, D["match_idx"][test_rows], a.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    n0 = D["n0"]
    if a.test:  # test outputs exist only when the test split was read
        if not a.test_frozen:  # the de_mirage layout of train_thgnn_allmaps.py
            p_dm = np.zeros(n0, dtype=np.float32)
            p_dm[test_rows] = p_test
            np.savez(out_dir / f"{tag}_pred.npz", p=p_dm, split=D["split"][:n0], match_idx=D["match_idx"][:n0],
                     round=D["round"][:n0], second=D["second"][:n0], match_keys=D["keys"][:int(D["match_idx"][:n0].max()) + 1])
        pl.DataFrame({"match_key": D["keys"][D["match_idx"][test_rows]], "round": D["round"][test_rows].astype(np.int64),
                      "second": D["second"][test_rows].astype(np.int64), "label": D["label"][test_rows].astype(np.int64),
                      "p": p_test.astype(np.float64)}).write_parquet(out_dir / f"{tag}_test_preds.parquet")
    torch.save(best_state, out_dir / f"{tag}.pt")
    metrics = {"tag": tag, "model": "graph transformer (player and round tokens, map-graph hop bias)", "graph": a.graph,
               "seed": a.seed, "scope": "all seven maps (train, validation); " + ("frozen test, all seven maps" if a.test_frozen else "de_mirage test"), "epochs": a.epochs,
               "steps_budget": steps, "batch": a.batch, "lr": a.lr, "eval_every": a.eval_every,
               "architecture": {"d_model": D_MODEL, "heads": HEADS, "layers": LAYERS, "ffn": FFN, "max_hop": MAX_HOP,
                                "map_nodes": K, "latent": LATENT, "gru_hidden": GRU_H},
               "clip": EPS, "history": history, "selected_step": best_step, "selection": "lowest validation log loss",
               "val_log_loss": best_val, "val_ci95_match_bootstrap": val_ci, "val_rows": int(val_rows.size),
               "val_matches": val_matches, "test_read": bool(a.test), "test_log_loss": test_ll,
               "test_ci95_match_bootstrap": test_ci, "test_rows": int(test_rows.size) if a.test else None,
               "test_matches": test_matches, "train_rows": int(train_rows.size),
               "device": str(device), "seconds": round(time.time() - t0)}
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("val_log_loss", "test_log_loss", "test_ci95_match_bootstrap", "test_rows",
                                               "test_matches", "selected_step")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
