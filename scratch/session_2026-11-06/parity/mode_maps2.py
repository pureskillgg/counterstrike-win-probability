"""2026 dataset size by map and mode, using the match headers (map_name, is_wingman) for every match, not the grid.

Reads rounds.parquet, grid.parquet and coverage_missing.csv (laptop scratch copy) and the header parquet on E:.
Writes per_match_headers.parquet and mode_maps2_summary.json next to this script. Nothing is written to S3.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
RND = r"G:\pureskill\csds-cs2\data\combined\rounds.parquet"
GRID = r"G:\pureskill\csds-cs2\data\combined\grid.parquet"
RAW = r"E:\csds_final\csds"
POOL = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]
WIN_LO, WIN_HI = "2026/10/09", "2026/10/25"

R = (pl.read_parquet(RND, columns=["match_key", "round", "n_t", "n_ct", "era", "use", "round_status"])
     .filter(pl.col("era") == "2026"))
mks = R["match_key"].unique().to_list()
print(f"2026 rounds {R.height}, matches {len(mks)}", flush=True)

cache = os.path.join(HERE, "per_match_headers.parquet")
if os.path.exists(cache):
    H = pl.read_parquet(cache)
else:
    def hdr(mk):
        p = mk.split("/")
        f = os.path.join(RAW, p[1], p[2], p[3], p[4], "header")
        try:
            t = pq.read_table(f).to_pylist()[0]
        except Exception:
            return (mk, "READ_ERROR", None, None, None)
        return (mk, t.get("map_name"), t.get("is_wingman"), t.get("max_unique_players"), t.get("match_type"))

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(hdr, mks))
    H = pl.DataFrame(rows, schema=["match_key", "map_name", "is_wingman", "max_unique_players", "match_type"],
                     orient="row")
    H.write_parquet(cache)
print("headers read:", H.height, "| read errors:", H.filter(pl.col("map_name") == "READ_ERROR").height, flush=True)

G = (pl.read_parquet(GRID, columns=["match_key", "era", "map_name"]).filter(pl.col("era") == "2026")
     .unique(subset=["match_key"]).rename({"map_name": "grid_map"}))
agree = H.join(G, on="match_key", how="inner")
print("grid matches compared:", agree.height, "| map agrees with header:", agree.filter(pl.col("map_name") == pl.col("grid_map")).height, flush=True)

M = R.group_by("match_key").agg((pl.col("n_t") + pl.col("n_ct")).max().alias("maxsum"))
RH = (R.join(H, on="match_key", how="left").join(M, on="match_key", how="left")
      .with_columns(pl.col("is_wingman").fill_null(False),
                    (((pl.col("n_t") == 5) & (pl.col("n_ct") <= 5)) | ((pl.col("n_ct") == 5) & (pl.col("n_t") <= 5))).alias("std"),
                    ((pl.col("n_t") == 5) & (pl.col("n_ct") == 5)).alias("fv"),
                    pl.col("match_key").str.split("/").list.get(1).alias("y"),
                    (pl.col("match_key").str.split("/").list.get(2) + "/" + pl.col("match_key").str.split("/").list.get(3)).alias("md"))
      .with_columns((pl.col("y") + "/" + pl.col("md")).alias("day")))
gap = pl.read_csv(os.path.join(HERE, "coverage_missing.csv")).select("match_key")

out = {}
out["all_2026_rounds"] = RH.height
out["use_true_rounds"] = RH.filter(pl.col("use")).height
out["use_true_matches"] = RH.filter(pl.col("use"))["match_key"].n_unique()
print("all 2026 rounds:", RH.height, "| use True:", out["use_true_rounds"], "| matches with use:", out["use_true_matches"], flush=True)

ut = RH.filter(pl.col("use"))
by_wing = ut.group_by("is_wingman").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches")).sort("is_wingman").rows()
out["use_by_wingman"] = by_wing
print("use True by is_wingman (rounds, matches):", by_wing, flush=True)

nw = ut.filter(~pl.col("is_wingman"))
out["use_nonwing_rounds"] = nw.height
print("use True, non-wingman rounds:", nw.height, flush=True)

fv = nw.filter(pl.col("fv"))
out["use_nonwing_5v5_rounds"] = fv.height
out["use_nonwing_5v5_matches"] = fv["match_key"].n_unique()
print("use True, non-wingman, 5v5 rounds:", fv.height, "matches:", fv["match_key"].n_unique(), flush=True)

by_map = (fv.group_by("map_name").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches"))
          .sort("rounds", descending=True))
print("by map (5v5, use, non-wing):", by_map.rows(), flush=True)
out["by_map_all"] = by_map.rows()

pool = fv.filter(pl.col("map_name").is_in(POOL))
out["pool_rounds"] = pool.height
out["pool_matches"] = pool["match_key"].n_unique()
print("active duty pool, 5v5, use, non-wing:", pool.height, "matches:", pool["match_key"].n_unique(), flush=True)
print("pool by map:", pool.group_by("map_name").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches"))
      .sort("rounds", descending=True).rows(), flush=True)

win = pool.filter((pl.col("day") >= WIN_LO) & (pl.col("day") <= WIN_HI))
win_cov = win.join(gap, on="match_key", how="anti")
out["pool_window_rounds"] = win.height
out["pool_window_covered_rounds"] = win_cov.height
out["pool_window_covered_matches"] = win_cov["match_key"].n_unique()
print("pool, Oct 9-25 window:", win.height, "rounds | covered by AWS status:", win_cov.height, "rounds,",
      win_cov["match_key"].n_unique(), "matches", flush=True)
print("covered by map:", win_cov.group_by("map_name").agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches"))
      .sort("rounds", descending=True).rows(), flush=True)

nonpool = fv.filter(~pl.col("map_name").is_in(POOL))
out["nonpool_5v5_rounds"] = nonpool.height
out["nonpool_5v5_matches"] = nonpool["match_key"].n_unique()
print("non-pool 5v5 use rounds:", nonpool.height, "matches:", nonpool["match_key"].n_unique(),
      nonpool.group_by("map_name").agg(pl.len().alias("rounds")).sort("rounds", descending=True).rows(), flush=True)

with open(os.path.join(HERE, "mode_maps2_summary.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
print("wrote mode_maps2_summary.json", flush=True)
