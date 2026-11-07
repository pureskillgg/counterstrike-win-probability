import glob, os
import numpy as np
import polars as pl
BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
TICK = 64.0
def load(ch, cols):
    files = sorted(glob.glob(os.path.join(BASE, f"basic_{ch}.COMBINED", "dataframe_*")))
    lf = pl.scan_parquet(files).filter(pl.col("match_key").str.starts_with("csds/2026/"))
    return lf.select(["match_key"] + cols).collect()
rend = load("round_end", ["round","tick","player_count"])
clean0 = rend.filter(pl.col("player_count")==10).select(["match_key","round",pl.col("tick").alias("end_tick")])
mk = clean0.select("match_key").unique().sort("match_key").with_row_index("midx")
clean = clean0.join(mk, on="match_key")
def in_clean(df):
    return df.join(clean.select(["match_key","round","end_tick","midx"]), on=["match_key","round"], how="inner").drop("match_key")
pk = in_clean(load("item_pickup", ["round","tick","item","def_index"]))
pk = pk.with_columns((pl.col("tick")-pl.col("end_tick")).alias("after"))
post = pk.filter(pl.col("after")>0)
print("pickups total in clean", pk.height, "after end", post.height)
print("after-end pickup offset in seconds pct 10/50/90/99:", [round(float(np.percentile(post["after"].to_numpy(),q))/64,2) for q in [10,50,90,99]])
print("by item (top):", post.group_by("item").agg(pl.len().alias("n")).sort("n", descending=True).head(10).to_dicts())
# are they at round end or in the next round's freeze? check next round start
rs = in_clean(load("round_start", ["round","tick"])).select(["midx","round","tick"]).rename({"tick":"rs_tick"})
rs_all = load("round_start", ["round","tick"]).select(["match_key","round","tick"])
rnext = rs_all.with_columns((pl.col("round")-1).alias("prev")).select(["match_key","prev","tick"]).rename({"prev":"round","tick":"next_start"})
pj = post.join(clean.select(["midx","match_key","round"]), on=["midx","round"], how="left")
print("pickups after end: sample rows")
print(post.head(5))
# check for pickups after end that are also before next round start (i.e., within same round in the sense of ticks)
gaps = pk.group_by(["midx","round"]).agg(pl.col("tick").max().alias("max_tick"), pl.col("end_tick").first()).with_columns(((pl.col("max_tick")-pl.col("end_tick"))/64).alias("max_after_s"))
print("max pickup tick minus end, seconds pct 50/90/99:", [round(float(np.percentile(gaps["max_after_s"].to_numpy(),q)),1) for q in [50,90,99]])
# round_end tick vs round_start of next round
print("rounds total", clean.height)
