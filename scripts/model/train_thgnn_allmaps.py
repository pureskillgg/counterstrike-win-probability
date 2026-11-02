"""THGNN-lite on all seven maps: one model, each second scored on its own map's graph, tested on the frozen de_mirage test seconds.

Same model as train_thgnn_lite.py (player encoder, player-to-map messages, three GAT layers over the map graph, readout, GRU over a
20-second window, head), with three changes for several maps:
  - each map keeps its own 48-node k-means graph (<map>_ext_mapgraph.npz) and a learned base embedding per map; a one-hot map
    indicator is part of the readout input;
  - training uses the train split of every map, validation uses the validation split of every map (best checkpoint, as in the
    single-map model), and the test is the frozen de_mirage test seconds only (the rows the 57-input baseline is scored on);
  - the per-second features (21 per slot: 6 state, 12 loadout, 3 velocity) are held on the GPU in float16, about 13 GB for the 31 million
    seconds, and the model arithmetic is float32. Previous-row indices stay on the host.

Pre-registered (written before any run): the all-maps THGNN beats the 57-input baseline if the mean paired log-loss difference on the
de_mirage test seconds (THGNN minus baseline) is <= -0.0005 and the 95% match-cluster CI upper bound is < 0. One seed, one budget of
50,000 optimizer steps (batch 512, about one pass over the all-maps training split). The checkpoint is the one with the lowest validation
log loss, checked every 10,000 steps.

Run in the torch environment, with the 3080 otherwise idle:
  G:/pureskill/csds-cs2/.pixi/envs/torch/python.exe -I -X utf8 -u scripts/model/train_thgnn_allmaps.py
Writes data/temporal/allmaps_thgnn_lite_pred.npz (de_mirage rows only, same layout as the single-map predictions, so
compare_paired_de_mirage.py scores it) and data/temporal/allmaps_thgnn_lite_metrics.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_temporal_round_model as ttr  # noqa: E402  (build_table, previous_rows, log_loss)
import train_temporal_allmaps as T  # noqa: E402  (MAPS, CACHE)

MAPS = T.MAPS
CACHE = T.CACHE
DATA, OUT, MODELS = ttr.DATA, ttr.OUT, ttr.MODELS
TRAIN, VAL, TEST = ttr.TRAIN, ttr.VAL, ttr.TEST
LOOKBACK = 20
N_SLOTS = 10
FP = 21  # 6 state + 12 loadout + 3 velocity
NODES = 48
G_DIM = 3 + 2 + len(MAPS)  # round state (3), alive fraction per side (2), map one-hot (7)
CHUNK = 500_000


class BatchedGAT(nn.Module):
    """Single-head GAT over a per-graph adjacency: each graph in the batch can have its own map graph."""

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Linear(dim, 1, bias=False)
        self.a_dst = nn.Linear(dim, 1, bias=False)

    def forward(self, M, A):
        # M (G, K, D), A (G, K, K) bool with self loops
        Wh = self.W(M)
        e = nn.functional.leaky_relu(self.a_src(Wh) + self.a_dst(Wh).transpose(1, 2), 0.2)
        e = e.masked_fill(~A, -1e9)
        alpha = torch.softmax(e, dim=-1) * A.float()
        return nn.functional.elu(alpha @ Wh)


class THGNNAll(nn.Module):
    def __init__(self, fp=FP, K=NODES, n_maps=len(MAPS), gfeat=G_DIM, hidden=64, latent=64, gru_hidden=64):
        super().__init__()
        self.K = K
        self.player_enc = nn.Sequential(nn.Linear(fp, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.p2m = nn.Linear(hidden, hidden)
        self.map_base = nn.Parameter(torch.zeros(n_maps, K, hidden))
        self.gat = nn.ModuleList([BatchedGAT(hidden) for _ in range(3)])
        self.readout = nn.Sequential(nn.Linear(3 * hidden + gfeat, latent), nn.ReLU())
        self.gru = nn.GRU(latent, gru_hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(gru_hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, Pw, alive_w, assign_w, G_w, A_w, base_idx, step_mask):
        # Pw (B, T, 10, fp); alive_w (B, T, 10); assign_w (B, T, 10) long, local node ids; G_w (B, T, gfeat);
        # A_w (B*T, K, K) bool; base_idx (B, T) long, map id per step; step_mask (B, T)
        B, T_ = alive_w.shape[:2]
        h = self.player_enc(Pw) * alive_w.unsqueeze(-1)  # (B, T, 10, H)
        ct = h[:, :, :5].sum(dim=2)
        tt = h[:, :, 5:].sum(dim=2)
        H = h.shape[-1]
        msg = self.p2m(h) * alive_w.unsqueeze(-1)
        M = torch.zeros(B, T_, self.K, H, device=h.device, dtype=h.dtype)
        M.scatter_add_(2, assign_w.unsqueeze(-1).expand(-1, -1, -1, H), msg)
        M = M + self.map_base[base_idx]  # (B, T, K, H): the map's own base embedding
        M = M.reshape(B * T_, self.K, H)
        for layer in self.gat:
            M = M + layer(M, A_w)
        map_pool = M.mean(dim=1).view(B, T_, H)
        z = self.readout(torch.cat([ct, tt, map_pool, G_w], dim=-1)) * step_mask.unsqueeze(-1)
        out, _ = self.gru(z)
        return self.head(out).squeeze(-1)  # (B, T) logits, oldest first


def load(device):
    """Every map's seconds in one set of device arrays. Rows are in MAPS order, so the de_mirage rows come first."""
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
    adj, keys = [], []
    key_offset, s = 0, 0
    for mi, m in enumerate(MAPS):
        arr, X, table, lab, st, use = ttr.build_table(m, f"{m}_ext", CACHE)
        n = X.shape[0]
        lo = np.load(DATA / f"{m}_ext_loadout.npy", mmap_mode="r")
        mg = np.load(DATA / f"{m}_ext_mapgraph.npz")
        A_m = mg["adjacency"].astype(bool)
        assert A_m.shape == (NODES, NODES), (m, A_m.shape)
        adj.append(A_m)
        am = mg["assign"].astype(np.int64)
        loc = ttr.previous_rows(arr["match_idx"], arr["round"], arr["second"], LOOKBACK)  # map-local row indices
        prev1 = loc[:, 1]
        has_prev = prev1 >= 0
        idx_prev = np.where(has_prev, prev1, 0)
        onehot = np.zeros((1, len(MAPS)), dtype=np.float32)
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
    return {"feat": feat, "G": G, "assign": assign, "A_all": torch.as_tensor(np.stack(adj), device=device),
            "label_t": torch.as_tensor(label, device=device), "usable_t": torch.as_tensor(usable, device=device),
            "map_t": torch.as_tensor(map_id, device=device),
            "label": label, "split": split, "usable": usable, "map_id": map_id, "round": round_id, "second": second,
            "match_idx": match_idx, "prev": prev, "keys": np.concatenate(keys), "N": N, "n0": int(lens[0])}


def log_loss(y, p):
    return ttr.log_loss(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_steps", type=int, default=50000, help="optimizer steps; the pre-registered budget is 50,000")
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--tag", default="allmaps_thgnn_lite")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda")
    print(f"device {device}; torch {torch.__version__}; seed {a.seed}; steps {a.max_steps}; batch {a.batch}", flush=True)
    t0 = time.time()
    D = load(device)
    print(f"rows {D['N']:,} loaded in {time.time() - t0:.0f} s; device {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    train_rows = np.flatnonzero((D["split"] == TRAIN) & D["usable"])
    val_rows = np.flatnonzero((D["split"] == VAL) & D["usable"])
    test_rows = np.flatnonzero((D["split"] == TEST) & D["usable"] & (D["map_id"] == 0))  # de_mirage test only
    print(f"train {train_rows.size:,}, val {val_rows.size:,} (all maps), test {test_rows.size:,} (de_mirage)", flush=True)

    def gather(rows):
        idx = D["prev"][rows]  # (B, T); k = 0 is the row itself, larger k further into the past
        mk_np = idx >= 0
        safe = np.where(mk_np, idx, 0)[:, ::-1].copy()  # oldest step first, as in train_thgnn_lite.py
        mk_np = mk_np[:, ::-1].copy()
        ix = torch.as_tensor(safe, dtype=torch.long, device=device)
        mk = torch.as_tensor(mk_np, dtype=torch.float32, device=device)
        Pw = D["feat"][ix].float() * mk[..., None, None]
        alive_w = (Pw[..., 0] > 0.5).float()
        assign_w = D["assign"][ix].long()
        G_w = D["G"][ix].float() * mk[..., None]
        used = D["usable_t"][ix] & (mk > 0.5)
        y = D["label_t"][ix]
        rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
        mapb = D["map_t"][rows_t]  # every step of a window is from the same match, hence the same map
        base_idx = mapb[:, None].expand(-1, LOOKBACK)
        A_w = D["A_all"][mapb][:, None].expand(-1, LOOKBACK, -1, -1).reshape(-1, NODES, NODES)
        return Pw, alive_w, assign_w, G_w, A_w, base_idx, mk, used, y

    model = THGNNAll().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    @torch.no_grad()
    def predict(rows, batch=2048):
        model.eval()
        out = np.empty(rows.size, dtype=np.float32)
        for s in range(0, rows.size, batch):
            r = rows[s:s + batch]
            Pw, alive_w, assign_w, G_w, A_w, base_idx, mk, _, _ = gather(r)
            logit = model(Pw, alive_w, assign_w, G_w, A_w, base_idx, mk)[:, -1]  # the current second is the newest step
            out[s:s + batch] = torch.sigmoid(logit).cpu().numpy()
        model.train()
        return out

    history = []
    best_val, best_step, best_state = float("inf"), 0, None
    order, pos = rng.permutation(train_rows), 0
    running, count = 0.0, 0.0
    t_train = time.time()

    def validate():
        nonlocal best_val, best_step, best_state
        v = log_loss(D["label"][val_rows], predict(val_rows))
        history.append({"step": step, "val_log_loss": v, "train_loss": running / max(count, 1.0),
                        "seconds": round(time.time() - t_train, 1)})
        print(json.dumps(history[-1]), flush=True)
        if v < best_val:
            best_val, best_step = v, step
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    step = 0
    while step < a.max_steps:
        if pos >= order.size:
            order, pos = rng.permutation(train_rows), 0
        r = order[pos:pos + a.batch]
        pos += a.batch
        Pw, alive_w, assign_w, G_w, A_w, base_idx, mk, used, y = gather(r)
        logit = model(Pw, alive_w, assign_w, G_w, A_w, base_idx, mk)
        per_step = bce(logit, y) * used.float()
        loss = per_step.sum() / used.float().sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += float(per_step.sum().item())
        count += float(used.float().sum().item())
        step += 1
        if step % 500 == 0:
            print(f"  step {step}/{a.max_steps}, train loss {running / max(count, 1.0):.4f}, {time.time() - t_train:.0f} s "
                  f"(device {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak)", flush=True)
        if step % a.eval_every == 0 or step == a.max_steps:
            validate()
            running, count = 0.0, 0.0

    model.load_state_dict(best_state)  # the lowest-validation checkpoint; the test set is never used to choose
    p_test = predict(test_rows)
    test_ll = log_loss(D["label"][test_rows], p_test)
    match = D["match_idx"][test_rows]
    uniq, inv = np.unique(match, return_inverse=True)
    yy, pp = D["label"][test_rows], np.clip(p_test, 1e-6, 1 - 1e-6)
    row_loss = -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
    sums, counts = np.zeros(uniq.size), np.zeros(uniq.size)
    np.add.at(sums, inv, row_loss)
    np.add.at(counts, inv, 1)
    brng = np.random.default_rng(a.seed)
    boot = np.array([sums[p].sum() / counts[p].sum() for p in (brng.integers(0, uniq.size, uniq.size) for _ in range(1000))])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # de_mirage rows only, in the single-map layout: they are the first n0 rows, so the existing paired script scores them unchanged
    n0 = D["n0"]
    p_dm = np.zeros(n0, dtype=np.float32)
    p_dm[test_rows] = p_test
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{a.tag}_pred.npz", p=p_dm, split=D["split"][:n0], match_idx=D["match_idx"][:n0],
             round=D["round"][:n0], second=D["second"][:n0], match_keys=D["keys"][:int(D["match_idx"][:n0].max()) + 1])
    torch.save(best_state, MODELS / f"{a.tag}.pt")
    metrics = {"tag": a.tag, "scope": "all seven maps (train, validation); de_mirage test", "seed": a.seed,
               "steps_budget": a.max_steps, "batch": a.batch, "history": history, "selected_step": best_step,
               "selection": "lowest validation log loss", "test_log_loss": test_ll,
               "test_ci95_match_bootstrap": [float(lo), float(hi)], "test_rows": int(test_rows.size),
               "test_matches": int(uniq.size), "train_rows": int(train_rows.size), "val_rows": int(val_rows.size),
               "seconds": round(time.time() - t0)}
    (OUT / f"{a.tag}_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("test_log_loss", "test_ci95_match_bootstrap", "test_rows", "test_matches",
                                               "selected_step")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
