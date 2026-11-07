import os
import glob
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds\basic_round_end.COMBINED"
ROUNDS = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"
PLAYERS = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"

entries = sorted(os.listdir(TOME))
print("entries:", entries)
pages = []
for name in entries:
    p = os.path.join(TOME, name)
    if os.path.isfile(p) and name.startswith("dataframe_"):
        pages.append(p)
    elif os.path.isdir(p):
        for f in sorted(glob.glob(os.path.join(p, "*"))):
            if os.path.isfile(f):
                pages.append(f)
print("page files:", pages)

frames = []
for p in pages:
    try:
        frames.append(pl.read_parquet(p))
    except Exception as e:  # noqa
        print("could not read as parquet:", p, repr(e)[:200])
if not frames:
    raise SystemExit("no pages read")

re_df = pl.concat(frames, how="diagonal_relaxed")
print("round_end schema:")
for k, v in re_df.collect_schema().items():
    print("   ", k, v)
print("total rows (all years):", re_df.height)
print("match_key prefixes:", re_df.select(pl.col("match_key").str.slice(0, 13)).unique().to_series().to_list()[:10])

re26 = re_df.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("2026 rows:", re26.height, "distinct match:", re26["match_key"].n_unique())
print("player_count values (2026):", re26.group_by("player_count").agg(pl.len().alias("n")).sort("player_count").to_dicts())
print("winner_team_code values (2026):", re26.group_by("winner_team_code").agg(pl.len().alias("n")).to_dicts())
with pl.Config(tbl_cols=20, tbl_width_chars=220):
    print(re26.head(5))

rounds = pl.read_parquet(ROUNDS).with_columns(pl.col("round").cast(pl.Int64))
print("rounds match_key sample:", rounds["match_key"][0])
re26j = re26.with_columns(pl.col("round").cast(pl.Int64))
cmp = rounds.join(
    re26j.select(["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"]),
    on=["match_key", "round"], how="left",
)
print("rounds rows:", cmp.height, " matched to round_end:", cmp.filter(pl.col("winner_team_code").is_not_null()).height)
cmp_m = cmp.filter(pl.col("winner_team_code").is_not_null())
cmp_m = cmp_m.with_columns(((pl.col("winner_team_code") == 3).cast(pl.Int64)).alias("ct_from_tome"))
agree = (cmp_m["ct_from_tome"] == cmp_m["ct_win"]).sum()
print("ct_win agrees with round_end winner:", int(agree), "of", cmp_m.height)
print("disagreements:")
with pl.Config(tbl_cols=20, tbl_width_chars=220):
    print(cmp_m.filter(pl.col("ct_from_tome") != pl.col("ct_win")).select(
        ["match_key", "round", "ct_win", "ct_from_tome", "winner_team_code", "player_count", "map_name", "split"]))
print("player_count among matched rounds:", cmp_m.group_by("player_count").agg(pl.len().alias("n")).to_dicts())
print("rounds with no round_end row:", cmp.filter(pl.col("winner_team_code").is_null()).height)
