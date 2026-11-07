"""Timing probe on three train-split bucket matches with about the average step count (72,567,700 steps / 57,811 matches).
Read-only on S3 and final/; scratch only. Compares STEP=2 (old) with STEP=1 (every second). Date filter: path date before 2026/09/01."""
import json
import os
import shutil
import sys
import time

sys.argv = ["probe"]
sys.path.insert(0, os.path.expanduser("~/adam/builder/fixes"))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pos_bucket_step as PB  # noqa: E402
import make_pos_features as MPF  # noqa: E402

v2keys = json.load(open(PB.FIN + "/export_clean_v2/match_keys.json"))
print("export_clean and export_clean_v2 match order identical:", v2keys == PB.keys, flush=True)
sp = np.load(PB.FIN + "/export_clean_v2/match_split.npy")
steps_by_mid = np.bincount(PB.rm, weights=PB.n_steps, minlength=len(PB.keys))
target = 72567700 / 57811
cands = [mk for mk in PB.bkeys & set(PB.kid) if int(sp[PB.kid[mk]]) == 0 and "/".join(mk.split("/")[1:4]) < "2026/09/01"]
cands.sort(key=lambda mk: abs(steps_by_mid[PB.kid[mk]] - target))
picked = cands[:3]
print("target steps per match", round(target), "| picked steps", [int(steps_by_mid[PB.kid[m]]) for m in picked], flush=True)
names = ["player_vector", "player_status", "player_info", "player_blind"]
tot = {2: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}
for mk in picked:
    folder = mk.rsplit("/", 1)[0]
    for step in (2, 1):
        PB.STEP = step
        shutil.rmtree(f"{PB.SCR}/{folder}", ignore_errors=True)
        t = time.time()
        for nm in names:
            PB.fetch(folder, nm)
        a = time.time() - t
        t = time.time()
        r = PB.one_match(mk)
        b = time.time() - t
        assert r[0] == "ok", r
        _, pos, sta, G, ev = r
        pxy_one = PB.pxy.filter(pl.col("match_key") == mk)
        t = time.time()
        f = MPF.feats_for_day(pos, sta, G, pxy_one, PB.sites, step, ev)
        c = time.time() - t
        for i, v in enumerate((a, b, c)):
            tot[step][i] += v
        print(f"  STEP={step} steps {G.height} pos rows {pos.height} feature rows {f.height} | download {a:.2f}s parse+join {b:.2f}s features {c:.2f}s", flush=True)
for step in (2, 1):
    a, b, c = tot[step]
    print(f"STEP={step} over 3 matches: download {a:.2f}s parse+join {b:.2f}s features {c:.2f}s | total {a + b + c:.2f}s | download share {100 * a / (a + b + c):.0f}%", flush=True)
