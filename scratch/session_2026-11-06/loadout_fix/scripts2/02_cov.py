import json, sys, collections
import numpy as np
import polars as pl

S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
OUT = S + "/out"
keys = json.load(open(S + "/match_keys.json"))
split = np.load(S + "/export/match_split.npy")
off = np.load(S + "/offsets.npy").astype(np.int64)
rm = np.load(S + "/round_match.npy").astype(np.int64)
rn = np.load(S + "/round_num.npy").astype(np.int64)
dates = [k.split("/")[1] + "-" + k.split("/")[2] + "-" + k.split("/")[3] for k in keys]
PROTECT_FROM = "2026-10-21"
mt = pl.DataFrame({"m": np.arange(len(keys), dtype=np.int64), "match_key": keys, "split": split.astype(np.int64),
                   "date": dates})
mt = mt.with_columns(
    ((pl.col("split") != 2) & (pl.col("date") < PROTECT_FROM)).alias("in_scope"),
    pl.col("date").str.slice(0, 7).alias("month"),
    pl.when(pl.col("date") >= PROTECT_FROM).then(pl.lit("protected_date")).when(pl.col("split") == 2).then(pl.lit("test_split")).otherwise(pl.lit("train_val")).alias("status"),
)
rounds_per_m = pl.DataFrame({"m": rm, "round": rn}).group_by("m").agg(pl.len().alias("export_rounds"))
mt = mt.join(rounds_per_m, on="m", how="left")
cov = pl.read_parquet(OUT + "/coverage_local.parquet")
cov = cov.with_columns(pl.col("m").cast(pl.Int64))
mt = mt.join(cov.select("m", "src", "rounds_with_rows", "steps_with_rows", "freeze_rows", "rows"), on="m", how="left")
mt = mt.with_columns(pl.col("src").fill_null("none"))
print("STATUS by split/date:")
print(mt.group_by("status").agg(pl.len().alias("matches")).sort("status"))
ins = mt.filter(pl.col("in_scope"))
print("IN SCOPE matches", ins.height)
print(ins.group_by(["month", "src"]).agg([pl.len().alias("matches"), pl.col("freeze_rows").fill_null(0).sum().alias("freeze_rows_sum"), pl.col("export_rounds").sum().alias("export_rounds")]).sort(["month", "src"]))
loc = ins.filter(pl.col("src") != "none")
print("in-scope with local rows", loc.height, "rounds covered with freeze row", int(loc["freeze_rows"].sum()), "of", int(loc["export_rounds"].sum()))
miss = ins.filter(pl.col("src") == "none")
print("in-scope with NO local rows", miss.height)
print(miss.group_by("month").agg(pl.len().alias("n")).sort("month"))
# per-match completeness among covered
cc = loc.with_columns((pl.col("freeze_rows") / pl.col("export_rounds")).alias("frac_rounds_with_freeze_row"))
print("covered matches with freeze_rows < export_rounds:", cc.filter(pl.col("freeze_rows") < pl.col("export_rounds")).height)
cc.write_parquet(OUT + "/match_local_cov.parquet")
mt.write_parquet(OUT + "/match_index.parquet")
