"""M3 extra 2: ceiling of a per-map offset, and per-map residuals by batch (2026 kept set).

Uses the out-of-fold predictions saved by M3_extra.py (scratchpad M3_oof.npz).
  1. In-sample ceiling: one logit offset per map fitted on all states of the no-map OOF logits.
     This is optimistic (it sees the states it is scored on). It bounds what a per-map prior
     could gain. Match-cluster bootstrap CI on the difference.
  2. Residuals (y - p) of the no-map OOF predictions: overall by batch, and per map by batch,
     to see which map offsets are stable across batches.
ASSUMPTION: batch = match_date 2026-10-08 versus other dates (see M3_extra.py).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M3_extra2.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
B = 2000
EPS = 1e-15
T0 = time.time()

S = pl.read_parquet(SCRATCH / "M3_states.parquet").sort("row_idx")
MT = pl.read_parquet(SCRATCH / "M3_matches.parquet")
oof = np.load(SCRATCH / "M3_oof.npz")
p_nm, p_mp = oof["p_nm"], oof["p_mp"]
y = S["label_ct_win"].to_numpy().astype(np.int64)
N = len(y)
mkeys = S["match_key"].to_list()
maps_list = S["map_name"].to_list()
match_list = sorted(set(mkeys))
mcode = {m: i for i, m in enumerate(match_list)}
mc = np.array([mcode[m] for m in mkeys], dtype=np.int64)
NM = len(match_list)
map_names = sorted(set(maps_list))
map_idx = {mp: i for i, mp in enumerate(map_names)}
ms_code = np.array([map_idx[m] for m in maps_list], dtype=np.int64)
J = len(map_names)
match_date = dict(zip(MT["match_key"].to_list(), MT["match_date"].to_list()))
batch_match = np.array(["A_2026-10-08" if match_date[m].startswith("2026-10-08") else "B_other"
                        for m in match_list])
state_batch = batch_match[mc]


def ll(yv, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(yv * np.log(p) + (1 - yv) * np.log(1 - p))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_offsets(z, yv, mcodes, J, iters=30):
    b = np.zeros(J)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(z + b[mcodes])))
        g = np.bincount(mcodes, weights=(yv - p), minlength=J)
        h = np.bincount(mcodes, weights=p * (1 - p), minlength=J)
        b += np.where(h > 1e-9, g / np.maximum(h, 1e-9), 0.0)
    return b


def boot_diff(la, lb, codes, seed=0):
    u, inv = np.unique(codes, return_inverse=True)
    n = np.bincount(inv).astype(float)
    sa = np.bincount(inv, weights=la, minlength=len(u))
    sb = np.bincount(inv, weights=lb, minlength=len(u))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    nn = n[idx].sum(1)
    d = sb[idx].sum(1) / nn - sa[idx].sum(1) / nn
    return float(sb.sum() / n.sum() - sa.sum() / n.sum()), float(np.percentile(d, 2.5)), \
        float(np.percentile(d, 97.5))


def boot_mean(vals, codes, seed=0):
    u, inv = np.unique(codes, return_inverse=True)
    n = np.bincount(inv).astype(float)
    s = np.bincount(inv, weights=vals, minlength=len(u))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    bm = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


l_nm = ll(y, p_nm)
z = logit(p_nm)
print("== 1. In-sample ceiling of a per-map logit offset (optimistic) ==")
b_all = fit_offsets(z, y, ms_code, J)
p_ceiling = 1.0 / (1.0 + np.exp(-(z + b_all[ms_code])))
l_ceil = ll(y, p_ceiling)
diff, lo, hi = boot_diff(l_nm, l_ceil, mc)
print(f"no map {l_nm.mean():.5f}; in-sample per-map offset {l_ceil.mean():.5f}; diff {diff:+.5f} "
      f"[{lo:+.5f}, {hi:+.5f}] (10 offsets fitted on all states; ceiling, not a test)")
print("offsets (logit) fitted in-sample: " + ", ".join(
    f"{map_names[j]} {b_all[j]:+.3f}" for j in np.argsort([-np.sum(ms_code == j) for j in range(J)])))

print("\n== 2. No-map OOF residuals (y - p) by batch ==")
r = y - p_nm
for b_name in ("A_2026-10-08", "B_other"):
    mb = state_batch == b_name
    m, lo_, hi_ = boot_mean(r[mb], mc[mb])
    print(f"batch {b_name}: states {int(mb.sum())}, mean residual {m:+.4f} [{lo_:+.4f}, {hi_:+.4f}], "
          f"observed CT {y[mb].mean():.4f}, mean predicted {p_nm[mb].mean():.4f}")

print("\nper map: residual by batch (A = 2026-10-08, B = other), with match-cluster 95% CI")
print(f"{'map':<11}{'A resid':>10}{'A CI':>22}{'B resid':>10}{'B CI':>22}{'A-B':>9}")
for j in np.argsort([-np.sum(ms_code == j) for j in range(J)]):
    cells = {}
    for b_name in ("A_2026-10-08", "B_other"):
        mm = (ms_code == j) & (state_batch == b_name)
        if mm.sum() < 50:
            cells[b_name] = None
            continue
        cells[b_name] = boot_mean(r[mm], mc[mm])
    a, bb = cells["A_2026-10-08"], cells["B_other"]
    fa = f"{a[0]:+.4f} [{a[1]:+.4f},{a[2]:+.4f}]" if a else "n<50"
    fb = f"{bb[0]:+.4f} [{bb[1]:+.4f},{bb[2]:+.4f}]" if bb else "n<50"
    diffab = f"{a[0] - bb[0]:+.4f}" if (a and bb) else "n/a"
    print(f"{map_names[j]:<11}{(a[0] if a else float('nan')):>+10.4f}{fa:>22}"
          f"{(bb[0] if bb else float('nan')):>+10.4f}{fb:>22}{diffab:>9}")
print(f"\ndone in {time.time() - T0:.1f}s")
