"""verify D7-2 (05): robustness of the axis comparison beyond the 3,287 first-after-swap rows.
Sets: (a) all 2024 rows with w_ok not null (any round, any swap status); (b) all round-13 rows with w_ok not null.
Axes: header match_date (UTC), csds path day, header build_num. Realizable single-threshold errors (splits only at value changes),
both orientations. ASSUMPTION: w_ok is the D7 ledger-vs-winner label; flip = not w_ok.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
LOG = ROOT / "logs" / "swarm2"
H = (pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num"])
     .rename({"key": "match_key"})
     .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=True).alias("mdt_h")))
X = pl.read_parquet(str(LOG / "D7_rounds_with_ledger_dp.parquet"), columns=["match_key", "round", "w_ok", "mdt", "key_day"]) \
    .filter(pl.col("w_ok").is_not_null()) \
    .join(H.select(["match_key", "build_num"]), on="match_key", how="left") \
    .with_columns(pl.col("key_day").str.replace_all("/", "-").str.to_date("%Y-%m-%d").cast(pl.Datetime("us")).alias("t_path"),
                  pl.col("mdt").alias("t_match"),
                  (~pl.col("w_ok")).alias("flip"))
print("rows with w_ok not null:", X.height, "| flips:", X.filter(pl.col("flip")).height, "| distinct matches:", X["match_key"].n_unique())
print("flips by build x round==13:", X.group_by(["build_num", pl.col("round") == 13, "flip"]).len().sort(["build_num", "round", "flip"]).rows())
pre = X.filter(pl.col("flip") & (pl.col("t_match") < __import__("datetime").datetime(2024, 2, 7, 1, 1, 18)))
print("flips before switch (any round):", pre.height, "| by build:", pre.group_by("build_num").len().rows())
print("flips before switch, sample:", pre.sort("t_match").select(["match_key", "round", "t_match", "build_num"]).head(5).with_columns(pl.col("match_key").str.slice(0, 40)))
print("flips by round (build 9961, any):", X.filter((pl.col("build_num") == 9961) & pl.col("flip")).group_by("round").len().sort("round").rows())
print("non-flip rows in build 9961:", X.filter((pl.col("build_num") == 9961) & (~pl.col("flip"))).height)

def realizable(vals, flips):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    v = [vals[i] for i in order]; f = [1 if flips[i] else 0 for i in order]
    n = len(f); Fb = [0] * (n + 1)
    for i in range(n): Fb[i + 1] = Fb[i] + f[i]
    Ft = Fb[n]; best_a = None; best_b = None
    for c in range(n + 1):
        if not (c == 0 or c == n or v[c - 1] < v[c]): continue
        a = Fb[c] + ((n - c) - (Ft - Fb[c]))
        b = (c - Fb[c]) + (Ft - Fb[c])
        lo = v[c - 1] if c > 0 else None; hi = v[c] if c < n else None
        if best_a is None or a < best_a[0]: best_a = (a, lo, hi)
        if best_b is None or b < best_b[0]: best_b = (b, lo, hi)
    return best_a, best_b

for label, D in [("(a) all rounds", X), ("(b) round 13 only", X.filter(pl.col("round") == 13))]:
    print(f"\n== {label}: rows={D.height} flips={D.filter(pl.col('flip')).height} matches={D['match_key'].n_unique()}")
    for axis in ["t_match", "t_path", "build_num"]:
        vals = D[axis].to_list(); flips = D["flip"].to_list()
        a, b = realizable(vals, flips)
        print(f"  axis={axis:<10} flip-after errors={a[0]} (last-before={a[1]}, first-after={a[2]}) | flip-before errors={b[0]}")
