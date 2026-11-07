"""D7-1 verifier, part F: (F1) best single-cut error count on round-13 bomb rows, match-time axis vs path-day axis;
(F2) mismatch counts by round number and build for all bomb-outcome rows (player_count==20).
ASSUMPTION: a single threshold cut; error = mismatches below the cut + clean rows at/above the cut.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path("C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_1_rounds.parquet"))
B = R.filter(pl.col("bomb_side").is_not_null() & (pl.col("player_count") == 20))
r13 = B.filter(pl.col("round") == 13)

def best_cut(df, axis):
    d = df.sort(axis)
    vals = d[axis].to_list(); y = d["mm"].cast(pl.Int8).to_list()
    n = len(y); best = None
    # cut positions only between distinct axis values
    for c in range(0, n + 1):
        if c > 0 and c < n and vals[c - 1] == vals[c]:
            continue
        err = sum(y[:c]) + sum(1 - v for v in y[c:])
        if best is None or err < best[0]:
            best = (err, c)
    err, c = best
    return err, (vals[c - 1] if c > 0 else None), (vals[c] if c < n else None), n

print("F1 round-13 bomb rows (n=%d, mismatches=%d)" % (r13.height, int(r13["mm"].sum())))
e_t, lo_t, hi_t, n = best_cut(r13, "mdt")
print(f"F1 match-time axis (UTC match_date): best single-cut errors={e_t}; last before cut={lo_t}; first at/after cut={hi_t}")
r13k = r13.with_columns(pl.col("key_day").alias("kd"))
e_k, lo_k, hi_k, _ = best_cut(r13k, "kd")
print(f"F1 path-day axis (csds key day): best single-cut errors={e_k}; last before cut={lo_k}; first at/after cut={hi_k}")
print("F1 path-day cut detail: rows with key_day >= 2024/02/12:", r13.filter(pl.col("key_day") >= "2024/02/12").height,
      "clean among them:", r13.filter((pl.col("key_day") >= "2024/02/12") & (~pl.col("mm"))).height)

print("\nF2 bomb-outcome rows (player_count==20): mismatches by round number and build (round: build9842 n/mm | build9961 n/mm)")
tab = (B.group_by(["round", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
         .sort(["round", "build_num"]))
out = []
for r in sorted(tab["round"].unique().to_list()):
    s = tab.filter(pl.col("round") == r)
    a = s.filter(pl.col("build_num") == 9842)
    b = s.filter(pl.col("build_num") == 9961)
    out.append((r, (a["n"][0], a["mm"][0]) if a.height else (0, 0), (b["n"][0], b["mm"][0]) if b.height else (0, 0)))
for row in out:
    print("  round", row[0], "| 9842 n,mm:", row[1], "| 9961 n,mm:", row[2])
print("F2 totals build 9842:", B.filter(pl.col("build_num") == 9842).select(pl.len(), pl.col("mm").sum()).rows(),
      "| build 9961:", B.filter(pl.col("build_num") == 9961).select(pl.len(), pl.col("mm").sum()).rows())
nonswap = B.filter((pl.col("build_num") == 9961) & (~pl.col("round").is_in([9, 13, 25, 28])))
print("F2 build 9961 bomb rows outside rounds 9/13/25/28:", nonswap.height, "mismatches:", int(nonswap["mm"].sum()))
