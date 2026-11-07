# A1: round_end per-match counts vs header final scores (t_starters + ct_starters), all 1752 matches, in-tome only.
import os, glob, collections, json
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"),
                    columns=["key", "match_id", "max_rounds", "t_starters_score_final", "ct_starters_score_final", "match_date", "match_type", "map_name"])
h = h.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("S"))
pages = sorted(glob.glob(os.path.join(T, "basic_round_end.2026-09-18,2026-10-08", "dataframe_*")))
re_ = pl.concat([pl.scan_parquet(p) for p in pages], how="diagonal_relaxed").collect()
print("round_end rows", re_.height, "cols", re_.columns)
print("round values min/max:", re_.select(pl.col("round").min()).item(), re_.select(pl.col("round").max()).item())
print("round_end distinct match_key:", re_.get_column("match_key").n_unique())
print("round_end winner_team_code dist:", re_.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("round_end win_reason_code dist:", re_.group_by("win_reason_code").len().sort("win_reason_code").rows())
dup = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("duplicate (match_key, round) groups:", dup.height, "rows in dup groups:", int(dup.get_column("len").sum()) if dup.height else 0)
agg = re_.group_by("match_key").agg([
    pl.len().alias("rows"), pl.col("round").n_unique().alias("distinct_rounds"),
    pl.col("round").max().alias("max_round"), pl.col("round").min().alias("min_round")])
m = h.join(agg, left_on="key", right_on="match_key", how="left").with_columns([
    pl.col("rows").fill_null(0), pl.col("distinct_rounds").fill_null(0)])
print("matches with zero round_end rows:", m.filter(pl.col("rows") == 0).height)
m = m.with_columns((pl.col("rows") - pl.col("S")).alias("diff_rows"), (pl.col("distinct_rounds") - pl.col("S")).alias("diff_distinct"))
print("diff_rows (rows - S) distribution:", m.group_by("diff_rows").len().sort("diff_rows").rows())
print("diff_distinct distribution:", m.group_by("diff_distinct").len().sort("diff_distinct").rows())
print("rows == S:", m.filter(pl.col("rows") == pl.col("S")).height, " distinct == S:", m.filter(pl.col("distinct_rounds") == pl.col("S")).height, " of", m.height)
print("max_round vs S (max_round - S):", m.group_by((pl.col("max_round") - pl.col("S")).alias("d")).len().sort("d").rows())
print("by max_rounds: rows==S", m.group_by("max_rounds").agg(pl.col("diff_rows").eq(0).sum().alias("eq"), pl.len().alias("n")).rows())
mis = m.filter(pl.col("diff_distinct") != 0).select(["key", "match_id", "max_rounds", "S", "t_starters_score_final", "ct_starters_score_final", "rows", "distinct_rounds", "max_round", "min_round", "match_date", "match_type", "map_name"]).sort("key")
print("mismatch count (distinct != S):", mis.height)
mis.write_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_round_end_mismatch.csv")
print(mis.head(40).rows())
m.write_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_round_end_per_match.csv")
