import glob
import polars as pl

BASE = "G:/pureskill/csds-cs2/tomes/2024-02-g/tome/csds/"
SUF = ".2023-12-28,2024-02-25"


def read(ch, cols):
    files = sorted(glob.glob(BASE + f"basic_{ch}{SUF}/dataframe_*"))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


RE = read("round_end", ["match_key", "round", "player_count"])
print("2024 round_end player_count distribution:")
print(RE.group_by("player_count").agg(pl.len().alias("n")).sort("player_count"))
PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
per = PI.group_by(["match_key", "round"]).agg(pl.len().alias("slots"),
                                              (pl.col("team_code") == 2).sum().alias("nT"),
                                              (pl.col("team_code") == 3).sum().alias("nCT"))
j = RE.join(per, on=["match_key", "round"], how="left")
print("player_count x player_info slots (rows, 2024):")
print(j.group_by(["player_count", "slots"]).agg(pl.len().alias("n")).sort(["player_count", "slots"]).head(30))
print("player_count 20 rounds with exactly 5 T and 5 CT (player_info):",
      j.filter((pl.col("player_count") == 20) & (pl.col("nT") == 5) & (pl.col("nCT") == 5)).height)
print("player_count 10 rounds with exactly 5 T and 5 CT (player_info):",
      j.filter((pl.col("player_count") == 10) & (pl.col("nT") == 5) & (pl.col("nCT") == 5)).height)
print("team_code values in 2024 player_info:", PI["team_code"].unique().sort().to_list())
