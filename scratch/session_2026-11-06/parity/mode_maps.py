"""2026 rounds: side-count pattern (standard = one side has 5, the other has at most 5) and map/mode from match headers.

Read-only: rounds.parquet on the laptop, header parquet files on E:. Writes mode_maps_summary.json next to this script.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
RND = r"G:\pureskill\csds-cs2\data\combined\rounds.parquet"
RAW = r"E:\csds_final\csds"

R = pl.read_parquet(RND, columns=["match_key", "round", "n_t", "n_ct", "era", "use", "round_status"])
R = R.filter(pl.col("era") == "2026")
mks = R["match_key"].unique().to_list()
print(f"2026 rounds {R.height}, matches {len(mks)}", flush=True)


def hdr(mk):
    p = mk.split("/")
    f = os.path.join(RAW, p[1], p[2], p[3], p[4], "header")
    if not os.path.exists(f):
        return (mk, None, None, None, None, None)
    try:
        t = pq.read_table(f).to_pylist()[0]
    except Exception:
        return (mk, "READ_ERROR", None, None, None, None)
    return (mk, t.get("map_name"), t.get("is_wingman"), t.get("max_unique_players"), t.get("match_type"),
            t.get("max_rounds"))


with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(hdr, mks))
H = pl.DataFrame(rows, schema=["match_key", "map_name", "is_wingman", "max_unique_players", "match_type", "max_rounds"],
                 orient="row")
print("matches with no header on E:", H.filter(pl.col("map_name").is_null()).height, flush=True)

std = ((pl.col("n_t") == 5) & (pl.col("n_ct") <= 5)) | ((pl.col("n_ct") == 5) & (pl.col("n_t") <= 5))
RH = R.join(H, on="match_key", how="left").with_columns(std.alias("standard_sides"))

out = {}
out["rounds_total"] = RH.height
out["standard_sides_rounds"] = int(RH["standard_sides"].sum())
out["pattern_counts"] = RH.group_by(["n_t", "n_ct"]).agg(pl.len().alias("rounds")).sort("rounds", descending=True).head(15).rows()
print("standard side rounds:", out["standard_sides_rounds"], "of", RH.height, flush=True)
print("top patterns:", out["pattern_counts"][:10], flush=True)

print("\nmatch mode (headers, matches):", flush=True)
mode = H.group_by(["is_wingman", "match_type", "max_unique_players", "max_rounds"]).agg(pl.len().alias("matches")).sort("matches", descending=True)
print(mode.head(12), flush=True)
out["mode"] = mode.to_dicts()

print("\nmaps by matches:", flush=True)
mp = H.group_by("map_name").agg(pl.len().alias("matches")).sort("matches", descending=True)
print(mp, flush=True)
out["maps_matches"] = mp.to_dicts()

S = RH.filter(pl.col("standard_sides"))
mr = S.group_by("map_name").agg(pl.len().alias("standard_rounds")).sort("standard_rounds", descending=True)
print("\nmaps by standard-side rounds:", flush=True)
print(mr, flush=True)
out["maps_standard_rounds"] = mr.to_dicts()

top7 = mr.head(7)["map_name"].to_list()
t7 = S.filter(pl.col("map_name").is_in(top7))
print(f"\ntop 7 by standard-side rounds: {top7}", flush=True)
print(f"standard rounds in top 7: {t7.height} of {RH.height} total 2026 rounds ({t7.height / RH.height:.1%})", flush=True)
out["top7"] = top7
out["standard_in_top7"] = t7.height
with open(os.path.join(HERE, "mode_maps_summary.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
print("wrote mode_maps_summary.json", flush=True)
