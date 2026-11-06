"""Temporal win-probability model for one map: per-second set encoder, then a causal GRU over the last LOOKBACK seconds.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\torch\\python.exe -I -X utf8 scripts\\model\\train_temporal_round_model.py --map de_mirage

Inputs (read-only):
  data/round_state/<map>.npz        X (N, 10, 6) slot arrays, split, match_idx, round, second, match_keys
  data/round_state/<map>.json       scaling window
  data/combined/rounds.parquet      final: 2 = T, 3 = CT (label = 1 when CT wins)
  data/combined/cache_snapshot_2026.parquet  per-second state (bomb, round clock); columns checked at start
Model, per second: a per-slot MLP masked by the alive flag, summed within each team, then a small MLP to a vector.
Then a causal GRU reads the last LOOKBACK seconds of those vectors plus a few state numbers, and outputs
P(CT wins) at the current second. Rows are not shuffled within a round, and the window never looks ahead.
Training uses the train split; the checkpoint with the lowest validation log loss is kept (once per epoch, or every --eval_every steps). Test predictions are written for the comparison.
Outputs: data/temporal/<map>_gru_pred.npz (p, split, match_idx, round, second), data/temporal/<map>_gru_metrics.json,
models/temporal/<map>_gru.pt. Nothing is written under tomes/ or data/combined/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "round_state"
OUT = ROOT / "data" / "temporal"
MODELS = ROOT / "models" / "temporal"
TRAIN, VAL, TEST = 0, 1, 2
N_SLOTS, SLOTS_PER_SIDE = 10, 5
STATE_COLS = ["bomb_planted", "seconds_since_round_live", "seconds_since_plant"]  # snapshot columns, checked at start
CLOCK_SCALE, PLANT_SCALE = 115.0, 41.0  # bomb timer 41 s, round clock 115 s (pinned final model design)


def build_table(map_name, stem=None, cache="cache_snapshot_2026.parquet"):
    """stem: the round-state file (default the map name); cache: the snapshot cache whose state columns are joined on."""
    arr = np.load(DATA / f"{stem or map_name}.npz")
    cache_path = ROOT / "data" / "combined" / cache
    X = arr["X"]
    keys = arr["match_keys"]
    table = pl.DataFrame({
        "match_key": keys[arr["match_idx"]],
        "round": arr["round"].astype(np.int64),
        "second": arr["second"].astype(np.int64),
        "split": arr["split"].astype(np.int64),
    })
    rounds = (pl.read_parquet(ROOT / "data" / "combined" / "rounds.parquet", columns=["match_key", "round", "final"])
              .with_columns(pl.col("round").cast(pl.Int64))
              .unique(["match_key", "round"], keep="first"))
    snap_schema = pl.read_parquet_schema(cache_path)
    missing = [c for c in STATE_COLS if c not in snap_schema]
    if missing:
        raise SystemExit(f"snapshot is missing state columns {missing}; pick the right names before running")
    snap = (pl.read_parquet(cache_path,
                            columns=["match_key", "round", "second"] + STATE_COLS)
            .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
            .unique(["match_key", "round", "second"], keep="first"))
    table = (table.join(rounds, on=["match_key", "round"], how="left", maintain_order="left")
             .join(snap, on=["match_key", "round", "second"], how="left", maintain_order="left"))
    # every array below is indexed by position, so the joined table must stay in the round-state row order
    assert (table["round"].to_numpy() == arr["round"]).all() and (table["second"].to_numpy() == arr["second"]).all(), \
        "joins changed the row order of the round-state arrays"
    # Rows with no winner or no snapshot state are excluded from training and evaluation (they stay in the sequence windows
    # as zeros). The snapshot holds no rows at all for these rounds, probably the roster-defect rounds that the evaluation
    # rules exclude; this is counted, not guessed around.
    usable = (table["final"].is_not_null() & table[STATE_COLS[0]].is_not_null()).to_numpy()
    print(f"rows excluded from training and evaluation (no winner or no snapshot state): {int((~usable).sum())}", flush=True)
    label = (table["final"].fill_null(0).to_numpy() == 3).astype(np.float32)
    state = np.stack([
        table[STATE_COLS[0]].fill_null(0).cast(pl.Float32).to_numpy(),
        np.clip(table[STATE_COLS[1]].fill_null(0).cast(pl.Float32).to_numpy() / CLOCK_SCALE, 0, 2),
        np.clip(table[STATE_COLS[2]].fill_null(0).cast(pl.Float32).to_numpy() / PLANT_SCALE, 0, 2),
    ], axis=1).astype(np.float32)
    return arr, X, table, label, state, usable


def previous_rows(match_idx, round_id, second, lookback):
    """For each row, the indices of the same match's same round at second-k (k = 0..lookback-1), or -1 when not present.
    The key includes the match: without it, rows of other matches with the same round and second would be picked up."""
    n = round_id.size
    assert round_id.max() < 128 and second.max() < 1024, "key packing assumes round < 128 and second < 1024"
    key = ((match_idx.astype(np.int64) * 128 + round_id.astype(np.int64)) * 1024) + second.astype(np.int64)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    out = np.full((n, lookback), -1, dtype=np.int64)
    for k in range(lookback):
        target = key - k
        pos = np.searchsorted(skey, target)
        pos_c = np.minimum(pos, skey.size - 1)
        found = (pos < skey.size) & (skey[pos_c] == target)
        out[:, k] = np.where(found, order[pos_c], -1)
    return out


class GATBlock(nn.Module):
    """Per-second attention over the ten players (a single-head GAT with self-loops and alive masking)."""

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Linear(dim, 1, bias=False)
        self.a_dst = nn.Linear(dim, 1, bias=False)

    def forward(self, h, alive):
        # h (B, K, 10, D), alive (B, K, 10)
        Wh = self.W(h)
        e = nn.functional.leaky_relu(self.a_src(Wh) + self.a_dst(Wh).transpose(-1, -2), 0.2)  # (B, K, 10, 10)
        mask = alive.unsqueeze(-2) > 0  # neighbour j must be alive
        e = e.masked_fill(~mask, -1e9)
        alpha = torch.softmax(e, dim=-1) * mask.float()  # rows with no alive neighbour get zero weight
        out = alpha @ Wh
        return nn.functional.elu(out) * alive.unsqueeze(-1)


class TemporalRoundModel(nn.Module):
    def __init__(self, in_dim=6, latent=32, hidden=64, state_dim=3, gat=False):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.gat = GATBlock(hidden) if gat else None
        self.team = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.gru = nn.GRU(latent + state_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, Xw, Sw, mask):
        # Xw (B, K, 10, 6) oldest first; Sw (B, K, S); mask (B, K) 1 where a real second exists
        alive = Xw[..., :1]
        h = self.phi(Xw) * alive
        if self.gat is not None:
            h = h + self.gat(h, Xw[..., 0])
        ct = h[:, :, :SLOTS_PER_SIDE].sum(dim=2)
        t = h[:, :, SLOTS_PER_SIDE:].sum(dim=2)
        z = self.team(torch.cat([ct, t], dim=-1)) * mask.unsqueeze(-1)
        out, _ = self.gru(torch.cat([z, Sw * mask.unsqueeze(-1)], dim=-1))
        return self.head(out[:, -1]).squeeze(-1)


def gather_window(X_t, S_t, prev, rows, device):
    idx = prev[rows]  # (B, K) with -1 for missing
    mask = torch.as_tensor(idx >= 0, dtype=torch.float32, device=device)
    safe = torch.as_tensor(np.where(idx >= 0, idx, 0), dtype=torch.long, device=device)
    Xw = X_t[safe] * mask[..., None, None]
    Sw = S_t[safe]
    # reverse so the oldest second comes first (k is "seconds ago")
    return Xw.flip(1), Sw.flip(1), mask.flip(1)


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loadout", action="store_true", help="add the per-slot loadout features (health, armour, equipment)")
    ap.add_argument("--gat", action="store_true", help="add a per-second attention layer over the ten players before the GRU")
    ap.add_argument("--stem", default=None, help="round-state file stem (default: the map name), e.g. de_mirage_ext")
    ap.add_argument("--cache", default="cache_snapshot_2026.parquet", help="snapshot cache whose state columns are joined on")
    ap.add_argument("--suffix", default="", help="appended to the output names, so a control run does not overwrite a result")
    ap.add_argument("--seed", type=int, default=0, help="torch and numpy seed (initialisation, batch order)")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (0 = run all epochs)")
    ap.add_argument("--eval_every", type=int, default=0, help="validate every this many steps (0 = once per epoch)")
    a = ap.parse_args()
    stem = a.stem or a.map
    tag = ("_gat" if a.gat else "") + ("_loadout" if a.loadout else "") + "_gru" + a.suffix
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; torch {torch.__version__}", flush=True)

    arr, X, table, label, state, usable = build_table(a.map, stem, a.cache)
    if a.loadout:
        X = np.concatenate([X, np.load(DATA / f"{stem}_loadout.npy")], axis=2)
        print(f"added loadout features: X shape {X.shape}", flush=True)
    split = arr["split"].astype(np.int64)
    prev = previous_rows(arr["match_idx"], arr["round"], arr["second"], a.lookback)
    assert (prev[:, 0] == np.arange(prev.shape[0])).all(), "the k=0 window entry must be the row itself"
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    S_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(label, dtype=torch.float32, device=device)

    train_rows = np.flatnonzero((split == TRAIN) & usable)
    val_rows = np.flatnonzero((split == VAL) & usable)
    test_rows = np.flatnonzero((split == TEST) & usable)
    print(f"rows {X.shape[0]}; train {train_rows.size}, val {val_rows.size}, test {test_rows.size}", flush=True)

    model = TemporalRoundModel(in_dim=X.shape[2], state_dim=state.shape[1], gat=a.gat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def predict(rows):
        model.eval()
        out = np.empty(rows.size, dtype=np.float32)
        for s in range(0, rows.size, 8192):
            r = rows[s:s + 8192]
            Xw, Sw, m = gather_window(X_t, S_t, prev, r, device)
            out[s:s + 8192] = torch.sigmoid(model(Xw, Sw, m)).cpu().numpy()
        model.train()
        return out

    best = (np.inf, None)
    history = []
    t0 = time.time()
    step, last_eval = 0, 0

    def validate(record):
        nonlocal best, last_eval
        v = log_loss(label[val_rows], predict(val_rows))
        history.append({**record, "val_log_loss": v, "seconds": round(time.time() - t0, 1)})
        print(json.dumps(history[-1]), flush=True)
        last_eval = step
        if v < best[0]:
            best = (v, {k: t.detach().clone() for k, t in model.state_dict().items()})

    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(train_rows)
        running, count = 0.0, 0
        for s in range(0, order.size, a.batch):
            r = order[s:s + a.batch]
            Xw, Sw, m = gather_window(X_t, S_t, prev, r, device)
            logit = model(Xw, Sw, m)
            loss = loss_fn(logit, y_t[torch.as_tensor(r, dtype=torch.long, device=device)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * r.size
            count += r.size
            step += 1
            if (s // a.batch) % 250 == 0:
                steps = -(-order.size // a.batch)
                print(f"  epoch {epoch}: step {s // a.batch + 1}/{steps}, train loss {running / count:.4f}, "
                      f"{time.time() - t0:.0f} s", flush=True)
            if a.eval_every and step % a.eval_every == 0:
                validate({"epoch": epoch, "step": step})
            if a.max_steps and step >= a.max_steps:
                break
        if not a.eval_every:
            validate({"epoch": epoch, "train_loss": running / count})
        if a.max_steps and step >= a.max_steps:
            break
    if a.eval_every and last_eval != step:
        validate({"epoch": epoch, "step": step})

    model.load_state_dict(best[1])
    p_all = predict(np.arange(X.shape[0]))
    p_test = p_all[test_rows]
    test_ll = log_loss(label[test_rows], p_test)

    # match-cluster bootstrap on the test rows (same resampling as the internal baseline)
    match = arr["match_idx"][test_rows]
    uniq, inv = np.unique(match, return_inverse=True)
    yy, pp = label[test_rows], np.clip(p_test, 1e-6, 1 - 1e-6)
    row_loss = -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
    sums = np.zeros(uniq.size)
    counts = np.zeros(uniq.size)
    np.add.at(sums, inv, row_loss)
    np.add.at(counts, inv, 1)
    boot_rng = np.random.default_rng(0)
    boot = np.array([sums[p].sum() / counts[p].sum() for p in (boot_rng.integers(0, uniq.size, uniq.size) for _ in range(1000))])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{stem}{tag}_pred.npz", p=p_all, split=split, match_idx=arr["match_idx"],
             round=arr["round"], second=arr["second"], match_keys=arr["match_keys"])
    torch.save(best[1], MODELS / f"{stem}{tag}.pt")
    metrics = {"map": a.map, "stem": stem, "cache": a.cache, "loadout": a.loadout, "gat": a.gat, "seed": a.seed,
               "max_steps": a.max_steps, "eval_every": a.eval_every, "steps_run": step, "lookback_s": a.lookback,
               "epochs": a.epochs, "history": history,
               "test_log_loss": test_ll, "test_ci95_match_bootstrap": [float(lo), float(hi)],
               "test_rows": int(test_rows.size), "test_matches": int(uniq.size), "device": str(device)}
    (OUT / f"{stem}{tag}_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("test_log_loss", "test_ci95_match_bootstrap", "test_rows", "test_matches")}),
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
