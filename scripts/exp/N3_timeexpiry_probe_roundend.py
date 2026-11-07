import os, glob
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds\basic_round_end.COMBINED"
pages = sorted(glob.glob(os.path.join(TOME, "dataframe_*")))
print("pages:", len(pages))
frames = [pl.read_parquet(p) for p in pages]
re_df = pl.concat(frames, how="diagonal_relaxed")
print("schema:")
for k, v in re_df.collect_schema().items():
    print("   ", k, v)
print("total rows:", re_df.height)
re26 = re_df.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("2026 rows:", re26.height, "matches:", re26["match_key"].n_unique())
print("player_count:", re26.group_by("player_count").agg(pl.len().alias("n")).sort("player_count").to_dicts())
for col in ["win_reason_code", "win_reason_message", "winner_team_code"]:
    if col in re26.columns:
        print(col, re26.group_by(col).agg(pl.len().alias("n")).sort(col).to_dicts())
with pl.Config(tbl_cols=30, tbl_width_chars=260):
    print(re26.head(4))
g = pl.scan_parquet(r"G:\pureskill\csds-cs2\data\viz\per_tick_states.parquet")
gm = g.select(pl.col("match_key").unique()).collect()
print("grid matches:", gm.height)
rem = set(re26["match_key"].unique().to_list())
gmset = set(gm["match_key"].to_list())
print("grid matches in 2026 round_end:", len(gmset & rem), " grid-only:", len(gmset - rem))
p = pl.scan_parquet(r"G:\pureskill\csds-cs2\data\viz\per_tick_test_preds.parquet")
pm = p.select(pl.col("match_key").unique()).collect()
print("pred matches:", pm.height, " in grid:", len(set(pm["match_key"].to_list()) & gmset))
