# A1: characterize the 80 matches where round_end distinct rounds < header S. In-tome data only.
import os, glob
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
m = pl.read_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_round_end_per_match.csv", infer_schema_length=0).with_columns(
    pl.col("S").cast(pl.Int64), pl.col("rows").cast(pl.Int64), pl.col("distinct_rounds").cast(pl.Int64),
    pl.col("max_round").cast(pl.Int64), pl.col("min_round").cast(pl.Int64), pl.col("max_rounds").cast(pl.Int64))
mis = m.filter(pl.col("distinct_rounds") != pl.col("S"))
print("mismatch n", mis.height, "| contiguous 1..N (min=1 & max=distinct):", mis.filter((pl.col("min_round") == 1) & (pl.col("max_round") == pl.col("distinct_rounds"))).height)
print("gap S - rows distribution:", mis.group_by((pl.col("S") - pl.col("rows")).alias("gap")).len().sort("gap").rows())
print("by max_rounds:", mis.group_by("max_rounds").len().rows(), "| total by max_rounds:", m.group_by("max_rounds").len().rows())
print("by map (mismatch):", mis.group_by("map_name").len().sort("len", descending=True).rows())
print("by match_type (mismatch):", mis.group_by("match_type").len().rows(), "| all:", m.group_by("match_type").len().rows())
print("by key date (mismatch):", mis.with_columns(pl.col("key").str.slice(0, 15).alias("kd")).group_by("kd").len().sort("kd").rows())
print("mismatch rate by key date:", m.with_columns(pl.col("key").str.slice(0, 15).alias("kd"), (pl.col("distinct_rounds") != pl.col("S")).alias("bad")).group_by("kd").agg(pl.col("bad").sum().alias("bad"), pl.len().alias("n")).sort("kd").rows())
print("mismatch by match_date day:", mis.with_columns(pl.col("match_date").str.slice(0, 10).alias("d")).group_by("d").len().sort("d").rows())
# round_start / round_state / score_update coverage for these matches
rs = pl.concat([pl.scan_parquet(p) for p in sorted(glob.glob(os.path.join(T, "basic_round_start.2026-09-18,2026-10-08", "dataframe_*")))], how="diagonal_relaxed").select(["match_key", "round"]).collect()
rst = rs.group_by("match_key").agg(pl.col("round").max().alias("rs_max_round"), pl.col("round").n_unique().alias("rs_distinct"))
print("round_start rows per match == S (all):", (rs.group_by("match_key").len().join(m.select(["key","S"]), left_on="match_key", right_on="key").filter(pl.col("len") == pl.col("S")).height))
x = mis.join(rst, left_on="key", right_on="match_key", how="left")
print("mismatch: round_start max_round > round_end max_round:", x.filter(pl.col("rs_max_round") > pl.col("max_round")).height, "of", x.height)
print("mismatch: round_start distinct == S:", x.filter(pl.col("rs_distinct") == pl.col("S")).height)
print("mismatch: round_start rows gap distribution (rs_distinct - rows):", x.with_columns((pl.col("rs_distinct") - pl.col("rows")).alias("g")).group_by("g").len().sort("g").rows())
# round_state max round
rstate = pl.concat([pl.scan_parquet(p) for p in sorted(glob.glob(os.path.join(T, "basic_round_state.2026-09-18,2026-10-08", "dataframe_*")))], how="diagonal_relaxed").select(["match_key", "round"]).collect()
rst2 = rstate.group_by("match_key").agg(pl.col("round").max().alias("st_max_round"), pl.col("round").n_unique().alias("st_distinct"))
y = mis.join(rst2, left_on="key", right_on="match_key", how="left")
print("mismatch: round_state max_round > round_end max:", y.filter(pl.col("st_max_round") > pl.col("max_round")).height, "of", y.height)
print("mismatch: round_state distinct == S:", y.filter(pl.col("st_distinct") == pl.col("S")).height)
print("sample:", y.select(["key","S","rows","max_round","rs_distinct" if False else "S","st_max_round","st_distinct"]).head(8).rows())
