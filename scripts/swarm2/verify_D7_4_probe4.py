# Probe: which stream carries the -429 offset? Plant->explode interval (C4 timer, nominal 40 s = 2560 ticks at 64) by period/build,
# and round_state round_end tick vs round_end channel tick.
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
def scan(ch): return pl.scan_parquet(files(BASE / f"basic_{ch}{SUF}"))
H = pl.read_parquet(files(BASE / "header")[0], columns=["key", "match_date", "build_num", "map_name", "server_name", "tick_save_rate"]).rename({"key": "match_key"}) \
    .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))
bs = scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_planted", "bomb_exploded", "bomb_defused"])).select(["match_key", "round", "tick", "event_type"]).collect()
pl_ = bs.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("plant_tick"))
ex = bs.filter(pl.col("event_type") == "bomb_exploded").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("exp_tick"))
df = ex.join(pl_, on=["match_key", "round"], how="left").join(H, on="match_key", how="left") \
       .with_columns((pl.col("exp_tick") - pl.col("plant_tick")).alias("plant_to_exp"),
                     pl.when(pl.col("mdt") < pl.datetime(2024, 2, 7, 1, 1, 18)).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"))
print("exploded rounds:", df.height, "| with plant:", df.filter(pl.col("plant_tick").is_not_null()).height)
print("plant_to_exp by period (ticks):")
print(df.filter(pl.col("plant_to_exp").is_not_null()).group_by("period").agg(pl.len().alias("n"),
      pl.col("plant_to_exp").min().alias("min"), pl.col("plant_to_exp").median().alias("median"), pl.col("plant_to_exp").max().alias("max"),
      (pl.col("plant_to_exp") == 2560).sum().alias("n_eq_2560")).sort("period"))
print("plant_to_exp top values, post:", df.filter((pl.col("period") == "post") & pl.col("plant_to_exp").is_not_null()).group_by("plant_to_exp").len().sort("len", descending=True).head(6).rows())
print("plant_to_exp top values, pre:", df.filter((pl.col("period") == "pre") & pl.col("plant_to_exp").is_not_null()).group_by("plant_to_exp").len().sort("len", descending=True).head(6).rows())
print("plant_to_exp by build (top values):")
for b in sorted(df["build_num"].drop_nulls().unique().to_list()):
    sub = df.filter((pl.col("build_num") == b) & pl.col("plant_to_exp").is_not_null())
    print("  build", b, "n", sub.height, "top", sub.group_by("plant_to_exp").len().sort("len", descending=True).head(4).rows())
# round_state round_end vs round_end channel
re_ = scan("round_end").select(["match_key", "round", "tick"]).collect()
rse = scan("round_state").filter(pl.col("event_type") == "round_end").select(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("rs_end")).collect()
J = re_.join(rse, on=["match_key", "round"], how="inner").with_columns((pl.col("rs_end") - pl.col("tick")).alias("d"))
print("round_state round_end rows matched:", J.height, "| rs_end - round_end channel tick top:", J.group_by("d").len().sort("len", descending=True).head(5).rows())
print("build_num of header (first 3):", H.head(3).select(["match_key", "build_num", "server_name", "tick_save_rate"]).rows())
