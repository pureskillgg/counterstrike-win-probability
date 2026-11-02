"""S2 check: row coverage of the holdout inputs written under data/holdout_b/ (counts only; no labels, no model output)."""
import glob
import json
import os
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
HB = os.path.join(ROOT, "data", "holdout_b")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
grid = pl.read_parquet(os.path.join(HB, "grid.parquet"), columns=["match_key", "map_name", "second"]).group_by("map_name").agg(pl.len().alias("grid_rows"))
grid = {r["map_name"]: r["grid_rows"] for r in grid.iter_rows(named=True)}
print("map | grid rows | round_state rows | loadout slots missing status | geo rows")
tot = {"grid": 0, "rs": 0, "geo": 0}
for m in MAPS:
    lj = json.load(open(os.path.join(HB, "round_state", f"{m}_ext_loadout.json"), encoding="utf-8"))
    rj = json.load(open(os.path.join(HB, "round_state", f"{m}_ext.json"), encoding="utf-8"))
    geo_rows = pl.scan_parquet(os.path.join(HB, "geo", f"{m}_ext.parquet")).select(pl.len()).collect().item()
    g = grid[m]
    tot["grid"] += g; tot["rs"] += rj["X_shape"][0]; tot["geo"] += geo_rows
    print(f"{m} | {g:,} | {rj['X_shape'][0]:,} | {lj['position_rows_without_status']} | {geo_rows:,} | rs_stats rows_per_split {rj['stats']['rows_per_split']} | dropped sentinel {rj['stats']['rows_dropped_sentinel_pid']}, bad coords {rj['stats']['rows_dropped_bad_coordinates']}, days skipped {len(rj['days_skipped'])}")
print("totals", tot)
C = pl.read_parquet(os.path.join(HB, "cache_snapshot_2026_h.parquet"))
feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
print("cache rows", C.height, "matches", C["match_key"].n_unique(), "features present", sum(f in C.columns for f in feats), "of", len(feats))
nulls = {f: C[f].null_count() for f in feats if C[f].null_count()}
print("features with nulls in holdout cache:", nulls)
S = pl.scan_parquet(os.path.join(HB, "status_1hz", "*.parquet")).select(pl.len()).collect().item()
P = pl.scan_parquet(os.path.join(HB, "positions_1hz", "*.parquet")).select(pl.len()).collect().item()
print("status rows", S, "positions rows", P)
