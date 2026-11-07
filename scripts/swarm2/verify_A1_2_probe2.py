# verify A1-2: value ranges and id coverage for player_info / player_personal / score_update / round_end (2026 tome, lazy, projected).
import os, glob
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"

def chan(name):
    folder = [d for d in os.listdir(T) if d.startswith(name + ".")][0]
    return sorted(glob.glob(os.path.join(T, folder, "dataframe_*")))

pi = pl.scan_parquet(chan("basic_player_info")).select(["match_key", "round", "player_id", "team_code"]).collect()
pp = pl.scan_parquet(chan("basic_player_personal")).select(["match_key", "round", "player_id", "player_id_fixed", "steam_id", "is_bot"]).collect()
su = pl.scan_parquet(chan("basic_score_update")).select(["match_key", "round", "tick", "team_code", "old_score", "new_score"]).collect()
re_ = pl.scan_parquet(chan("basic_round_end")).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()

print("player_info rows:", pi.height, "| matches:", pi.get_column("match_key").n_unique())
print("player_info round range:", pi.get_column("round").min(), pi.get_column("round").max())
print("player_info team_code values:", pi.get_column("team_code").value_counts().sort("team_code").rows())
print("player_personal rows:", pp.height, "| is_bot:", pp.get_column("is_bot").value_counts().rows())
print("player_personal null steam_id:", pp.get_column("steam_id").null_count(), "| distinct steam_id:", pp.get_column("steam_id").n_unique())
print("player_personal round range:", pp.get_column("round").min(), pp.get_column("round").max())
print("score_update rows:", su.height, "| team_code values:", su.get_column("team_code").value_counts().sort("team_code").rows())
print("round_end rows:", re_.height, "| winner values:", re_.get_column("winner_team_code").value_counts().sort("winner_team_code").rows())
print("round_end round range:", re_.get_column("round").min(), re_.get_column("round").max())
print("player_info rows for one sample match:")
k = pi.get_column("match_key").head(1).item()
print(k)
sub = pi.filter(pl.col("match_key") == k)
print(sub.group_by("round").agg(pl.len().alias("n"), pl.col("team_code").value_counts().alias("tc")).sort("round").head(20))
print("player_personal for sample match:")
print(pp.filter(pl.col("match_key") == k).sort(["round", "player_id"]).head(12))
print("distinct (player_id) per match in player_info:", pi.filter(pl.col("match_key") == k).get_column("player_id").n_unique())
