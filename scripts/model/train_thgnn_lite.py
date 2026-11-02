"""THGNN-style reimplementation on our de_mirage data (a reverse-engineering test, not the paper's model).

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\torch\\python.exe -I -X utf8 -u scripts\\model\\train_thgnn_lite.py --map de_mirage

What follows the paper (Szmida and Toka, SSAC26):
- A heterogeneous snapshot per second: player nodes with state features, and map nodes joined to the players.
- Player-to-player self-loop encoder, then player-to-map messages (players sent to the nearest map node), then GAT
  convolutions over the map graph (three layers, dense attention over the adjacency with self loops).
- A recurrent layer over the window of snapshots, then a per-step output (probability of CT winning).
- Trained on round winners; evaluated per second on the frozen test split.

What differs, on purpose:
- The map graph is data-driven (build_map_graph.py: 48 k-means nodes from train positions), not hand-built.
- The window is 20 seconds at 1 Hz (the paper: 20 snapshots over 5 seconds, 4 Hz). Our positions are 1 Hz.
- The recurrent layer runs over pooled snapshot vectors (one GRU), not one GraphGRU per node type.
- Player features: positions, sin and cos yaw, velocity (one-second difference), health, armour, helmet, defuser,
  money, equipment value, defusing and grenade flags (from the loadout file). No HLTV history features (KAST, Impact,
  rating), because the data licence rules out rating-from-history features.
Outputs match train_temporal_round_model.py so compare_paired_de_mirage.py can score them on the same seconds:
data/temporal/<map>_thgnn_lite_pred.npz and data/temporal/<map>_thgnn_lite_metrics.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_temporal_round_model as ttr  # noqa: E402  (build_table, previous_rows)

ROOT = ttr.ROOT
DATA = ttr.DATA
OUT = ttr.OUT
MODELS = ttr.MODELS
TRAIN, VAL, TEST = ttr.TRAIN, ttr.VAL, ttr.TEST
LOOKBACK = 20
N_SLOTS = 10


class DenseGAT(nn.Module):
    """Single-head GAT over a fixed adjacency (K nodes), batched over graphs."""

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Linear(dim, 1, bias=False)
        self.a_dst = nn.Linear(dim, 1, bias=False)

    def forward(self, M, A):
        # M (G, K, D), A (K, K) bool with self loops
        Wh = self.W(M)
        e = nn.functional.leaky_relu(self.a_src(Wh) + self.a_dst(Wh).transpose(1, 2), 0.2)
        e = e.masked_fill(~A.unsqueeze(0), -1e9)
        alpha = torch.softmax(e, dim=-1) * A.unsqueeze(0).float()
        return nn.functional.elu(alpha @ Wh)


class THGNNLite(nn.Module):
    def __init__(self, fp, K, hidden=64, latent=64, gru_hidden=64, gfeat=5):
        super().__init__()
        self.K = K
        self.player_enc = nn.Sequential(nn.Linear(fp, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.p2m = nn.Linear(hidden, hidden)
        self.map_base = nn.Parameter(torch.zeros(K, hidden))
        self.gat = nn.ModuleList([DenseGAT(hidden) for _ in range(3)])
        self.readout = nn.Sequential(nn.Linear(3 * hidden + gfeat, latent), nn.ReLU())
        self.gru = nn.GRU(latent, gru_hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(gru_hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, Pw, alive_w, assign_w, G_w, A, step_mask):
        # Pw (B, T, 10, fp); alive_w (B, T, 10); assign_w (B, T, 10) long, clamped; G_w (B, T, gfeat); step_mask (B, T)
        B, T = alive_w.shape[:2]
        h = self.player_enc(Pw) * alive_w.unsqueeze(-1)  # (B, T, 10, H)
        ct = h[:, :, :5].sum(dim=2)
        tt = h[:, :, 5:].sum(dim=2)
        # players -> their nearest map node (scatter sum), then map base embedding
        H = h.shape[-1]
        msg = self.p2m(h) * alive_w.unsqueeze(-1)
        M = torch.zeros(B, T, self.K, H, device=h.device, dtype=h.dtype)
        M.scatter_add_(2, assign_w.unsqueeze(-1).expand(-1, -1, -1, H), msg)
        M = M + self.map_base.view(1, 1, self.K, H)
        M = M.reshape(B * T, self.K, H)
        for layer in self.gat:
            M = M + layer(M, A)
        map_pool = M.mean(dim=1).view(B, T, H)
        z = self.readout(torch.cat([ct, tt, map_pool, G_w], dim=-1)) * step_mask.unsqueeze(-1)
        out, _ = self.gru(z)
        return self.head(out).squeeze(-1)  # (B, T) logits, oldest first


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="thgnn_lite")
    ap.add_argument("--train_frac", type=float, default=1.0, help="share of train matches to use (learning curve); val and test unchanged")
    ap.add_argument("--stem", default=None, help="round-state file stem (default: the map name), e.g. de_mirage_ext")
    ap.add_argument("--cache", default="cache_snapshot_2026.parquet", help="snapshot cache whose state columns are joined on")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (0 = run all epochs)")
    ap.add_argument("--eval_every", type=int, default=0, help="validate every this many steps (0 = once per epoch)")
    ap.add_argument("--graph", default=None, help="map graph file (default: <stem>_mapgraph.npz), e.g. the navigation-mesh regions")
    ap.add_argument("--seed", type=int, default=0, help="torch and sampling seed (default 0, the seed of every earlier run)")
    a = ap.parse_args()
    stem = a.stem or a.map
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; torch {torch.__version__}", flush=True)

    arr, X, table, label, state, usable = ttr.build_table(a.map, stem, a.cache)
    loadout = np.load(DATA / f"{stem}_loadout.npy")
    mg = np.load(a.graph or (DATA / f"{stem}_mapgraph.npz"))
    assign_np = mg["assign"].astype(np.int64)
    A_np = mg["adjacency"]
    K = A_np.shape[0]
    split = arr["split"].astype(np.int64)
    N = X.shape[0]

    prev = ttr.previous_rows(arr["match_idx"], arr["round"], arr["second"], LOOKBACK)
    prev1 = prev[:, 1]
    # one-second velocity per slot from the previous row of the same round (zero when missing or dead)
    vel = np.zeros((N, N_SLOTS, 3), dtype=np.float32)
    has_prev = prev1 >= 0
    idx_prev = np.where(has_prev, prev1, 0)
    both = has_prev[:, None] & (X[:, :, 0] > 0.5) & (X[idx_prev, :, 0] > 0.5)
    vel[both] = (X[:, :, 1:4] - X[idx_prev, :, 1:4])[both]
    P_np = np.concatenate([X, loadout, vel], axis=2)  # (N, 10, 6+12+3), already float32; a second copy would double the peak memory
    assert P_np.dtype == np.float32, P_np.dtype
    fp = P_np.shape[2]
    G_np = np.concatenate([state, X[:, :5, 0].sum(1, keepdims=True) / 5.0, X[:, 5:, 0].sum(1, keepdims=True) / 5.0],
                          axis=1).astype(np.float32)

    P_t = torch.as_tensor(P_np, device=device)
    G_t = torch.as_tensor(G_np, device=device)
    assign_t = torch.as_tensor(np.maximum(assign_np, 0), dtype=torch.long, device=device)
    alive_t = torch.as_tensor(X[:, :, 0] > 0.5, dtype=torch.float32, device=device)
    label_t = torch.as_tensor(label, dtype=torch.float32, device=device)
    usable_t = torch.as_tensor(usable, device=device)
    prev_t = torch.as_tensor(prev, dtype=torch.long, device=device)
    A_t = torch.as_tensor(A_np, dtype=torch.bool, device=device)

    def gather(rows):
        r = torch.as_tensor(rows, dtype=torch.long, device=device)
        idx = prev_t[r]  # (B, T), k = 0 is the row itself, k increases into the past
        step_mask = (idx >= 0).float()
        safe = idx.clamp(min=0)
        idx_old_first = safe.flip(1)
        mask_old_first = step_mask.flip(1)
        Pw = P_t[idx_old_first] * mask_old_first[..., None, None]
        alive_w = alive_t[idx_old_first] * mask_old_first[..., None]
        assign_w = assign_t[idx_old_first]
        G_w = G_t[idx_old_first] * mask_old_first[..., None]
        used = usable_t[idx_old_first] & mask_old_first.bool()
        return Pw, alive_w, assign_w, G_w, mask_old_first, used, label_t[idx_old_first]

    model = THGNNLite(fp, K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    @torch.no_grad()
    def predict_rows(rows, batch=2048):
        model.eval()
        out = np.empty(rows.size, dtype=np.float32)
        for s in range(0, rows.size, batch):
            r = rows[s:s + batch]
            Pw, alive_w, assign_w, G_w, sm, _, _ = gather(r)
            logit = model(Pw, alive_w, assign_w, G_w, A_t, sm)[:, -1]  # the current second is the newest step
            out[s:s + batch] = torch.sigmoid(logit).cpu().numpy()
        model.train()
        return out

    train_rows = np.flatnonzero((split == TRAIN) & usable)
    if a.train_frac < 1.0:
        train_matches = np.unique(arr["match_idx"][train_rows])
        keep = np.random.default_rng(0).permutation(train_matches)[: int(round(a.train_frac * train_matches.size))]
        train_rows = train_rows[np.isin(arr["match_idx"][train_rows], keep)]
        print(f"train_frac {a.train_frac}: {keep.size} of {train_matches.size} train matches", flush=True)
    val_rows = np.flatnonzero((split == VAL) & usable)
    test_rows = np.flatnonzero((split == TEST) & usable)
    print(f"features per slot {fp}; map nodes {K}; train {train_rows.size}, val {val_rows.size}, test {test_rows.size}",
          flush=True)

    history = []
    best_val, best_epoch, best_state = float("inf"), 0, None
    t0 = time.time()
    step, last_eval = 0, 0

    def validate(record):
        nonlocal best_val, best_epoch, best_state, last_eval
        v = log_loss(label[val_rows], predict_rows(val_rows)[:])
        history.append({**record, "val_log_loss": v, "seconds": round(time.time() - t0, 1)})
        print(json.dumps(history[-1]), flush=True)
        last_eval = step
        if v < best_val:
            best_val, best_epoch = v, record.get("epoch", 0)
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(train_rows)
        running, count = 0.0, 0.0
        for s in range(0, order.size, a.batch):
            Pw, alive_w, assign_w, G_w, sm, used, y = gather(order[s:s + a.batch])
            logit = model(Pw, alive_w, assign_w, G_w, A_t, sm)
            per_step = bce(logit, y) * used.float()
            loss = per_step.sum() / used.float().sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(per_step.sum().item())
            count += float(used.float().sum().item())
            step += 1
            if a.eval_every and step % a.eval_every == 0:
                validate({"epoch": epoch, "step": step})
            if a.max_steps and step >= a.max_steps:
                break
        if not a.eval_every:
            validate({"epoch": epoch, "train_loss": running / max(count, 1.0)})
        if a.max_steps and step >= a.max_steps:
            break
    if a.eval_every and last_eval != step:
        validate({"epoch": epoch, "step": step})

    model.load_state_dict(best_state)  # the checkpoint with the lowest validation log loss; test is never used to choose
    p_all = predict_rows(np.arange(N))
    p_test = p_all[test_rows]
    test_ll = log_loss(label[test_rows], p_test)
    match = arr["match_idx"][test_rows]
    uniq, inv = np.unique(match, return_inverse=True)
    yy, pp = label[test_rows], np.clip(p_test, 1e-6, 1 - 1e-6)
    row_loss = -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
    sums = np.zeros(uniq.size)
    counts = np.zeros(uniq.size)
    np.add.at(sums, inv, row_loss)
    np.add.at(counts, inv, 1)
    brng = np.random.default_rng(a.seed)
    boot = np.array([sums[p].sum() / counts[p].sum() for p in (brng.integers(0, uniq.size, uniq.size) for _ in range(1000))])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{stem}_{a.tag}_pred.npz", p=p_all, split=split, match_idx=arr["match_idx"],
             round=arr["round"], second=arr["second"], match_keys=arr["match_keys"])
    torch.save(model.state_dict(), MODELS / f"{stem}_{a.tag}.pt")
    metrics = {"map": a.map, "stem": stem, "cache": a.cache, "model": "THGNN-lite (data-driven map graph, 20 s window at 1 Hz)", "epochs": a.epochs,
               "max_steps": a.max_steps, "eval_every": a.eval_every, "steps_run": step,
               "selected_epoch": best_epoch, "selection": "lowest validation log loss",
               "history": history, "test_log_loss": test_ll, "test_ci95_match_bootstrap": [float(lo), float(hi)],
               "test_rows": int(test_rows.size), "test_matches": int(uniq.size), "device": str(device)}
    (OUT / f"{stem}_{a.tag}_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("test_log_loss", "test_ci95_match_bootstrap", "test_rows", "test_matches")}),
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
