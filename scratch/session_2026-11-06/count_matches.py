"""Inventory: how many distinct matches does each local dataset cover? Reads only match identifiers, never per-second rows."""
import sys
from pathlib import Path
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2/data")

print("== round_state (per-map .npz files) ==")
rs_files = sorted(ROOT.glob("round_state/*.npz"))
print(f"npz files: {len(rs_files)}")
rs_all = set()
for f in rs_files:
    with np.load(f, allow_pickle=True) as z:
        if "match_keys" not in z.files:
            print(f"  {f.name}: no match_keys member; members = {z.files}")
            continue
        keys = z["match_keys"]
        uniq = set(map(str, keys.tolist() if hasattr(keys, "tolist") else keys))
    rs_all |= uniq
    print(f"  {f.name}: {len(uniq):,} distinct matches")
print(f"round_state total distinct matches (union across maps): {len(rs_all):,}")

print("\n== cs2data cache (combined/cache_snapshot_2026_ext.parquet) ==")
cache = ROOT / "combined" / "cache_snapshot_2026_ext.parquet"
n_cache = pl.scan_parquet(cache).select(pl.col("match_key").n_unique()).collect().item()
print(f"distinct matches: {n_cache:,}")

print("\n== site grid (diagnostics/allmaps_pos24_trainsites_grid) ==")
grid_all = set()
for f in sorted((ROOT / "diagnostics" / "allmaps_pos24_trainsites_grid").glob("*.parquet")):
    keys = pl.read_parquet(f, columns=["match_key"])["match_key"].unique().to_list()
    grid_all |= set(keys)
    print(f"  {f.name}: {len(keys):,} distinct matches")
print(f"site grid total distinct matches: {len(grid_all):,}")

print("\n== overlap ==")
print(f"round_state matches also in the cache: {len(rs_all & set(pl.read_parquet(cache, columns=['match_key'])['match_key'].unique().to_list())):,}")
