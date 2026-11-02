"""Walking-travel map control per snapshot (de_mirage, 2026, every 10th second), from awpy's reachability method.

For each snapshot (every living player's position at one second), awpy labels each navigation area "ct", "t", "contested" or "neutral"
by comparing each side's walking travel distance to the area (awpy.map_control.compute_map_control, method="reachability"; no visibility
mesh is needed). The features are the size-weighted fractions: ct_fraction, t_fraction, contested_fraction, and net_control = ct - t.

Writes data/diagnostics/map_control_de_mirage_stride10.parquet (match_key, round, second, mc_ct, mc_t, mc_contested, mc_net). Snapshots that
awpy cannot evaluate are counted in the printed summary and left null.
Run in the navigation environment (awpy 3.x):
  C:/Users/<user>/pureskill-nav312/Scripts/python.exe scripts/model/map_control_features.py
"""
import glob
import os
import sys
import time

import polars as pl
import awpy._awpy as core
import awpy.map_control as mc

ROOT = "G:/pureskill/csds-cs2"
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "map_control_de_mirage_stride10.parquet")
KEY = ["match_key", "round", "second"]
STRIDE = 10
CT, T = 3, 2


def main():
    t0 = time.time()
    nav = core.NavMesh(os.path.join(ROOT, "data", "geometry", "navs", "de_mirage.nav"))
    mt = pl.read_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet"), columns=["match_key", "map_name"])
    keys = mt.filter(pl.col("map_name") == "de_mirage")["match_key"].to_list()
    rows = []
    errors = 0
    total = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "2026-*.parquet"))):
        p = pl.read_parquet(f, columns=["match_key", "round", "second", "side", "x", "y", "z"])
        p = p.filter(pl.col("match_key").is_in(keys) & (pl.col("second") % STRIDE == 0) & pl.col("side").is_in([CT, T]))
        if p.height == 0:
            continue
        g = p.group_by(KEY, maintain_order=True).agg(pl.col("x"), pl.col("y"), pl.col("z"), pl.col("side"))
        out = {k: [] for k in KEY + ["mc_ct", "mc_t", "mc_contested", "mc_net"]}
        for r in g.iter_rows(named=True):
            total += 1
            players = [(float(x), float(y), float(z), "ct" if s == CT else "t", False, False, None)
                       for x, y, z, s in zip(r["x"], r["y"], r["z"], r["side"])]
            try:
                res = mc.compute_map_control(nav, players, method="reachability")
                vals = (res["ct_fraction"], res["t_fraction"], res["contested_fraction"], res["net_control"])
            except Exception:
                errors += 1
                vals = (None, None, None, None)
            for k in KEY:
                out[k].append(r[k])
            for k, v in zip(["mc_ct", "mc_t", "mc_contested", "mc_net"], vals):
                out[k].append(v)
        rows.append(pl.DataFrame(out).with_columns([pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)]))
        print(f"  {os.path.basename(f)}: {g.height:,} snapshots, {time.time() - t0:.0f} s", flush=True)
    R = pl.concat(rows).sort(KEY)
    R.write_parquet(OUT)
    print(f"wrote {OUT}: {R.height:,} snapshots, {errors:,} of {total:,} could not be evaluated, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
