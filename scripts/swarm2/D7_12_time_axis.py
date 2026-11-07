"""D7-12: which clock explains the flip? Compare best single-threshold error counts for first-after-swap flips
(rounds 9/13 with ledger side) using header match_date (UTC) versus the csds path day (key_day).
Also check that the first -429 lag match and the first flip are the same match.
ASSUMPTION: match_date is the true match time; key path day is the storage/ingest day (tested here, not assumed).
"""
import sys, datetime as dt
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
X = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_with_ledger_dp.parquet"))
L = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_dp.parquet"))
fas = L.filter(pl.col("swap") == True).with_columns((pl.col("round") + 1).alias("round")).select(["match_key", "round"]).with_columns(pl.lit(True).alias("fas"))
Y = X.join(fas, on=["match_key", "round"], how="left").with_columns(pl.col("fas").fill_null(False)).filter(pl.col("w_ok").is_not_null())
F = Y.filter(pl.col("fas") & pl.col("round").is_in([9, 13]))

def best_single_threshold(df, key):
    d = df.sort(key)
    y = d["w_ok"].cast(pl.Int8).to_list(); n = len(y)
    pf = [0]*(n+1); pa = [0]*(n+1)
    for i, v in enumerate(y):
        pf[i+1] = pf[i] + (1-v); pa[i+1] = pa[i] + v
    errs = [pf[c] + (pa[n]-pa[c]) for c in range(n+1)]
    c = min(range(n+1), key=lambda i: errs[i])
    return n, errs[c], d[key][c-1] if c > 0 else None, d[key][c] if c < n else None

for key in ["mdt", "key_day_dt"]:
    if key == "key_day_dt":
        F2 = F.with_columns(pl.col("key_day").str.replace_all("/", "-").str.to_datetime("%Y-%m-%d").alias("key_day_dt"))
    else:
        F2 = F
    n, e, lo, hi = best_single_threshold(F2, key)
    print(f"axis={key:<12} n={n} best single-threshold errors={e} | last agree before={lo} | first flip after={hi}")

# same-match check for the first lag -429 match and the first flip
lagm = (X.filter(pl.col("bomb_side").is_not_null()).with_columns((pl.col("out_tick") - pl.col("tick")).alias("lag"))
          .filter(pl.col("lag") == -429).sort("mdt").head(3).select(["mdt", "match_key", "round"]))
print("\nfirst bomb rounds with lag -429:"); print(lagm.with_columns(pl.col("match_key").str.slice(0,50)))
flip = F.filter(~pl.col("w_ok")).sort("mdt").head(3).select(["mdt", "match_key", "round"])
print("first fas flips:"); print(flip.with_columns(pl.col("match_key").str.slice(0,50)))
print("same match at first flip time:", flip["match_key"][0] == lagm["match_key"][0] if flip.height and lagm.height else None)
