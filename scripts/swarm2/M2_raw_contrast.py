"""M2 raw contrast: CT win rate by map vs rest of era, on RAW rounds (every round with a winner,
no bake-off filters), with the same match-cluster bootstrap as M2_analysis.py (2000 resamples,
seed 0). Compared with the kept-set contrast, the difference shows how much the bake-off filters
move map-level effects.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M2_raw_contrast.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
SCRATCH = Path("C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/"
               "e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
B = 2000
SEED = 0
MAPS = ["de_mirage", "de_dust2", "de_inferno", "de_ancient", "de_anubis", "de_nuke", "de_cache",
        "de_vertigo", "de_overpass", "cs_office"]


def contrast_ci(n_all, c_all, g, B, seed):
    rng = np.random.default_rng(seed)
    M = len(n_all)
    W = rng.multinomial(M, np.full(M, 1.0 / M), size=B).astype(np.float64)
    num_m = W @ (c_all * g)
    den_m = W @ (n_all * g)
    num_r = W @ (c_all * (1 - g))
    den_r = W @ (n_all * (1 - g))
    diff = num_m / den_m - num_r / den_r
    return np.nanpercentile(diff, [2.5, 97.5])


for era in ["2026", "2024"]:
    r = pl.read_parquet(SCRATCH / f"M2_rounds_{era}.parquet").filter(pl.col("winner").is_in([2, 3]))
    mc = (r.group_by(["match_key", "map_name"]).agg(pl.len().alias("n"), pl.col("label_ct_win").sum().alias("c"))
          .sort("match_key"))
    allm = mc.group_by("match_key").agg(pl.col("n").sum(), pl.col("c").sum()).sort("match_key")
    n_all = allm["n"].to_numpy().astype(np.float64)
    c_all = allm["c"].to_numpy().astype(np.float64)
    keys = allm["match_key"].to_list()
    tot_r, tot_c = n_all.sum(), c_all.sum()
    print(f"\n[{era}] RAW rounds {int(tot_r)}, CT win rate {tot_c / tot_r:.4f}")
    print(f"{'map':<12}{'raw rounds':>11}{'raw CT':>9}{'raw rest':>10}{'raw diff':>10}{'raw diff 95% CI':>22}")
    for mp in MAPS:
        sub = mc.filter(pl.col("map_name") == mp)
        if sub.height == 0:
            print(f"{mp:<12}  (no raw rounds)")
            continue
        n_m = int(sub["n"].sum())
        c_m = int(sub["c"].sum())
        mk = set(sub["match_key"].to_list())
        g = np.array([1.0 if k in mk else 0.0 for k in keys])
        rate = c_m / n_m
        rest = (tot_c - c_m) / (tot_r - n_m)
        if 0 < g.sum() < len(g):
            lo, hi = contrast_ci(n_all, c_all, g, B, SEED)
            ci_s = "{:+.4f}..{:+.4f}".format(lo, hi)
        else:
            ci_s = "n/a"
        print(f"{mp:<12}{n_m:>11}{rate:>9.4f}{rest:>10.4f}{rate - rest:>+10.4f}{ci_s:>22}")
