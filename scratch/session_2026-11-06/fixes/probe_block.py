"""Timing of one 60-match block at STEP=2 and STEP=1, the same stages as run_chunk in pos_bucket.py. Writes nothing (no chunk file).
Matches: train split (v2 index code 0), path date before 2026/09/01 (guard asserted). Read-only on S3 and final/."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.argv = ["probe"]
sys.path.insert(0, os.path.expanduser("~/adam/builder/fixes"))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pos_bucket_step as PB  # noqa: E402
import make_pos_features as MPF  # noqa: E402

kb = json.load(open(PB.FIN + "/export_clean_v2/match_keys.json"))
idx_b = {k: i for i, k in enumerate(kb)}
sp = np.load(PB.FIN + "/export_clean_v2/match_split.npy")
in_v2 = [m for m in PB.bkeys if m in idx_b]
only_v1 = [m for m in PB.bkeys if m in PB.kid and m not in idx_b]
print("bucket keys in export_clean_v2:", len(in_v2), "| bucket keys in export_clean only (not in v2):", len(only_v1), flush=True)
cands = sorted(m for m in in_v2 if int(sp[idx_b[m]]) == 0 and "/".join(m.split("/")[1:4]) < "2026/09/01")
rng = np.random.default_rng(0)
block = [cands[i] for i in rng.choice(len(cands), 60, replace=False)]
assert all("/".join(m.split("/")[1:4]) < "2026/09/01" and int(sp[idx_b[m]]) == 0 for m in block)
print("block: 60 train matches, path dates", min("/".join(m.split('/')[1:4]) for m in block), "to", max("/".join(m.split('/')[1:4]) for m in block), flush=True)
bm = set(block)
for step in (2, 1):
    PB.STEP = step
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(PB.one_match, block))
    t_fp = time.time() - t0
    ok = [r for r in res if r[0] == "ok"]
    pos = pl.concat([r[1] for r in ok], how="vertical_relaxed")
    sta = pl.concat([r[2] for r in ok], how="vertical_relaxed")
    G = pl.concat([r[3] for r in ok], how="vertical_relaxed")
    evs = [r[4] for r in ok if r[4] is not None]
    ev = pl.concat(evs, how="vertical_relaxed") if evs else None
    pxy_b = PB.pxy.filter(pl.col("match_key").is_in(list(bm)))
    t1 = time.time()
    f = MPF.feats_for_day(pos, sta, G, pxy_b, PB.sites, step, ev)
    t_f = time.time() - t1
    print(f"STEP={step}: ok {len(ok)}/60 | fetch+parse+join {t_fp:.1f}s | features {t_f:.1f}s | grid steps {G.height} | pos rows {pos.height} | feature rows {f.height}", flush=True)
