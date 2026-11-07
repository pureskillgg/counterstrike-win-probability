# verify A1-2 probe 4: roster size per match, is_wingman flag, max_rounds, and player_id->steam_id consistency.
import os, glob, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def chan(name):
    folder = [d for d in os.listdir(T) if d.startswith(name + ".")][0]
    return sorted(glob.glob(os.path.join(T, folder, "dataframe_*")))


h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"),
                    columns=["key", "max_rounds", "is_wingman", "match_type", "t_starters_score_final",
                             "ct_starters_score_final", "unique_steamids", "match_date"])
print("header is_wingman:", h.get_column("is_wingman").value_counts().rows())
print("header max_rounds x is_wingman:", h.group_by(["max_rounds", "is_wingman"]).agg(pl.len().alias("n")).sort(["max_rounds", "is_wingman"]).rows())
print("header unique_steamids distribution:", h.group_by("unique_steamids").agg(pl.len().alias("n")).sort("unique_steamids").rows())
print("header match_type:", h.get_column("match_type").value_counts().rows())

pp = pl.scan_parquet(chan("basic_player_personal")).select(["match_key", "round", "player_id", "steam_id", "is_bot"]).collect()
pi = pl.scan_parquet(chan("basic_player_info")).select(["match_key", "round", "player_id", "team_code"]).collect()

roster = pp.group_by("match_key").agg(pl.col("player_id").n_unique().alias("npid"), pl.col("steam_id").n_unique().alias("nsid"))
print("roster by distinct player_id per match:", roster.group_by("npid").agg(pl.len().alias("n")).sort("npid").rows())
print("roster by distinct steam_id per match:", roster.group_by("nsid").agg(pl.len().alias("n")).sort("nsid").rows())

per_round = pi.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
print("player_info rows per (match,round) distribution:", per_round.group_by("n").agg(pl.len().alias("m")).sort("n").rows())

m = roster.join(h.select(["key", "max_rounds", "is_wingman"]), left_on="match_key", right_on="key", how="left")
print("roster npid x max_rounds x is_wingman:", m.group_by(["npid", "max_rounds", "is_wingman"]).agg(pl.len().alias("n")).sort(["npid", "max_rounds"]).rows())

# player_id that maps to >1 steam_id within a match
bad = pp.group_by(["match_key", "player_id"]).agg(pl.col("steam_id").n_unique().alias("k")).filter(pl.col("k") > 1)
print("player_id with >1 steam_id in match:", bad.height, bad.head(5).rows())
# steam_id that maps to >1 player_id within a match
bad2 = pp.group_by(["match_key", "steam_id"]).agg(pl.col("player_id").n_unique().alias("k")).filter(pl.col("k") > 1)
print("steam_id with >1 player_id in match:", bad2.height, bad2.head(5).rows())
# is_bot presence
print("is_bot rows:", pp.get_column("is_bot").value_counts().rows())
