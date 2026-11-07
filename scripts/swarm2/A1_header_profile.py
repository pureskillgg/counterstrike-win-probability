# A1: profile header dataframe (1752 rows) - date distribution, scores, keys vs match_id, round counts.
import glob, os, collections, json
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"), columns=[
    "key","match_id","match_date","match_type","map_name","max_rounds","t_starters_score_final",
    "ct_starters_score_final","tick_rate","number_of_points","sharecode","demo_id","game","platform","freeze_time","round_time_defuse"])
print("rows", h.height, "cols", h.width)
print("match_date top:", h.group_by("match_date").len().sort("match_date").head(40).rows())
print("key date prefix:", h.with_columns(pl.col("key").str.slice(0,15).alias("p")).group_by("p").len().sort("p").rows())
print("match_type:", h.group_by("match_type").len().rows())
print("map_name:", h.group_by("map_name").len().sort("len", descending=True).rows())
print("max_rounds:", h.group_by("max_rounds").len().rows())
print("tick_rate:", h.group_by("tick_rate").len().rows(), "game:", h.group_by("game").len().rows(), "platform:", h.group_by("platform").len().rows())
print("nulls:", {c: h.get_column(c).null_count() for c in h.columns})
# key tail vs match_id
kt = h.with_columns(pl.col("key").str.split("/").list.get(3).alias("kmid"))
print("key uuid == match_id count:", kt.filter(pl.col("kmid") == pl.col("match_id")).height, "of", h.height)
print("key-uuid mismatch sample:", kt.filter(pl.col("kmid") != pl.col("match_id")).select("key","match_id").head(3).rows())
print("sharecode unique:", h.get_column("sharecode").n_unique(), "demo_id unique:", h.get_column("demo_id").n_unique())
print("score final T/CT sum stats:", h.select((pl.col("t_starters_score_final")+pl.col("ct_starters_score_final")).alias("s")).describe().rows())
print("score sum distribution:", h.group_by((pl.col("t_starters_score_final")+pl.col("ct_starters_score_final")).alias("s")).len().sort("s").rows())
