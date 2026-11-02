"""Learn a round-state embedding z (LATENT numbers per state) by next-state prediction, for one map.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\torch\\python.exe -I -X utf8 scripts\\model\\train_round_state_encoder.py --map de_mirage --horizon 3

Input: data/round_state/<map>.npz from build_round_state_map.py (X (N, 10, 6), split, match_idx, round, second).
Model: a permutation-invariant set encoder. A per-slot MLP, masked by the alive flag, is summed within each team
(slots 0-4 CT, 5-9 T); a small MLP maps the two team sums to z. A decoder maps z back to the ten slots.
Target: the positions of the living players HORIZON seconds later in the same round, slot by slot (the slot convention
is fixed within a round). The round winner and player identity are never used.
Loss: squared error over living target slots, in scaled coordinates. Trained on the train split, checked on val.
Baseline: persistence (predict the current positions). The encoder must beat it on val to have learned motion.
Outputs: data/round_state/<map>_h<H>.npz (z for every row, plus split, match_idx, round, second),
         data/round_state/<map>_h<H>_metrics.json, and models/round_state/<map>_h<H>.pt (weights).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "round_state"
MODELS = ROOT / "models" / "round_state"
TRAIN, VAL, TEST = 0, 1, 2
SLOTS_PER_SIDE = 5
N_SLOTS = 10
LATENT = 8


def load(map_name):
    npz = np.load(DATA / f"{map_name}.npz")
    return {k: npz[k] for k in npz.files}


def next_state_rows(a, horizon):
    """For each row, the row index of the same round HORIZON seconds later, or -1 if that second is not in the data."""
    m = a["match_idx"].astype(np.int64)
    r = a["round"].astype(np.int64)
    s = a["second"].astype(np.int64)
    assert r.max() < 128 and s.max() < 1024, "key packing assumes round < 128 and second < 1024"
    key = (m * 128 + r) * 1024 + s
    order = np.argsort(key, kind="stable")
    skey = key[order]
    target = key + horizon
    pos = np.searchsorted(skey, target)
    pos_c = np.minimum(pos, skey.size - 1)
    found = (pos < skey.size) & (skey[pos_c] == target)
    return np.where(found, order[pos_c], -1)


class SetEncoder(nn.Module):
    def __init__(self, latent=LATENT, hidden=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, N_SLOTS * 3))

    def encode(self, X):
        h = self.phi(X) * X[:, :, :1]  # dead or missing slots contribute nothing
        ct = h[:, :SLOTS_PER_SIDE].sum(dim=1)
        t = h[:, SLOTS_PER_SIDE:].sum(dim=1)
        return self.rho(torch.cat([ct, t], dim=1))

    def forward(self, X):
        # The decoder predicts the change from each player's current position. Predicting absolute positions through an
        # 8-number bottleneck was worse than repeating the current positions (persistence), so the change is what is learned.
        z = self.encode(X)
        return z, X[:, :, 1:4] + self.dec(z).view(-1, N_SLOTS, 3)


def masked_mse(pred, target, alive):
    se = ((pred - target) ** 2).sum(dim=-1)
    return (se * alive).sum() / alive.sum().clamp(min=1.0)


def world_rmse(pred, target, alive, window):
    """Root mean squared distance over living target slots, in world units (x and y share the xy span, z has its own)."""
    xy = window["xy_max"] - window["xy_min"]
    zs = window["z_max"] - window["z_min"]
    d = (pred - target) * torch.tensor([xy, xy, zs], device=pred.device)
    dist2 = (d ** 2).sum(dim=-1)
    return float(torch.sqrt((dist2 * alive).sum() / alive.sum().clamp(min=1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    a_ = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; torch {torch.__version__}", flush=True)

    a = load(a_.map)
    meta = json.loads((DATA / f"{a_.map}.json").read_text(encoding="utf-8"))
    window = meta["window"]
    tgt = next_state_rows(a, a_.horizon)
    usable = tgt >= 0
    X_np = a["X"]
    split = a["split"]
    print(f"rows {X_np.shape[0]}; with a target {int(usable.sum())} (horizon {a_.horizon} s)", flush=True)

    X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
    T_idx = torch.as_tensor(np.where(usable, tgt, 0), dtype=torch.long, device=device)
    U = torch.as_tensor(usable, device=device)
    S = torch.as_tensor(split, device=device)

    train_idx = np.flatnonzero(usable & (split == TRAIN))
    val_idx = np.flatnonzero(usable & (split == VAL))
    print(f"train pairs {train_idx.size}, val pairs {val_idx.size}", flush=True)

    def batch_tensors(idx):
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        cur = X[idx_t]
        tgt_rows = T_idx[idx_t]
        tgt_xyz = X[tgt_rows, :, 1:4]
        tgt_alive = X[tgt_rows, :, 0]
        return cur, tgt_xyz, tgt_alive

    model = SetEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a_.lr)

    @torch.no_grad()
    def evaluate(idx):
        model.eval()
        total_model, total_persist, total_world, n = 0.0, 0.0, 0.0, 0
        for s in range(0, idx.size, 65536):
            cur, tx, ta = batch_tensors(idx[s:s + 65536])
            _, pred = model(cur)
            persist = cur[:, :, 1:4]
            w = ta.sum().item()
            total_model += masked_mse(pred, tx, ta).item() * w
            total_persist += masked_mse(persist, tx, ta).item() * w
            total_world += world_rmse(pred, tx, ta, window) * w
            n += w
        model.train()
        return total_model / n, total_persist / n, total_world / n

    t0 = time.time()
    history = []
    for epoch in range(1, a_.epochs + 1):
        order = rng.permutation(train_idx)
        running, count = 0.0, 0
        for s in range(0, order.size, a_.batch):
            cur, tx, ta = batch_tensors(order[s:s + a_.batch])
            _, pred = model(cur)
            loss = masked_mse(pred, tx, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * ta.sum().item()
            count += ta.sum().item()
        v_model, v_persist, v_world = evaluate(val_idx)
        row = {
            "epoch": epoch,
            "train_mse": running / count,
            "val_mse": v_model,
            "val_persistence_mse": v_persist,
            "val_rmse_world": v_world,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    # embed every row (all splits) with the trained encoder
    model.eval()
    Z = np.empty((X_np.shape[0], LATENT), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, X_np.shape[0], 65536):
            cur = X[s:s + 65536]
            Z[s:s + 65536] = model.encode(cur).cpu().numpy()

    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    stem = f"{a_.map}_h{a_.horizon}"
    np.savez(DATA / f"{stem}.npz", z=Z, split=split, match_idx=a["match_idx"], round=a["round"], second=a["second"],
             match_keys=a["match_keys"])
    torch.save(model.state_dict(), MODELS / f"{stem}.pt")
    metrics = {"map": a_.map, "horizon_s": a_.horizon, "latent": LATENT, "epochs": a_.epochs, "history": history,
               "final_val": history[-1], "device": str(device), "window": window}
    (DATA / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    last = history[-1]
    verdict = "beats" if last["val_mse"] < last["val_persistence_mse"] else "does NOT beat"
    print(f"encoder {verdict} persistence on val: model {last['val_mse']:.5f} vs persistence {last['val_persistence_mse']:.5f}",
          flush=True)
    print(f"wrote {DATA / (stem + '.npz')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
