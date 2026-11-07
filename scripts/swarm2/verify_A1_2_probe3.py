# verify A1-2 probe 3: inspect player_personal / player_info structure for one claimed 16-round match.
import os, glob, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def chan(name):
    folder = [d for d in os.listdir(T) if d.startswith(name + ".")][0]
    return sorted(glob.glob(os.path.join(T, folder, "dataframe_*")))


pi =pl.scan_parquet(chan("basic_player_info")).select(["match_key", "round", "player_id", "team_code", "wins", "rank"]).collect()
pp = pl.scan_parquet(chan("basic_player_personal")).select(["match_key", "round", "player_id", "player_controller_id", "player_id_pawn", "name", "steam_id", "is_bot", "player_id_fixed"]).collect()

k = "csds/2026/10/01/m_36b246426c63e2e81c3165c3/csds"
print("== player_personal for match, rounds 1-2 ==")
print(pp.filter((pl.col("match_key") == k) & (pl.col("round") <= 2)).sort(["round", "player_id"]))
print("== player_personal distinct per match (count of rows per round) ==")
print(pp.filter(pl.col("match_key") == k).group_by("round").len().sort("round").head(20))
print("== player_info for match rounds 1-3 ==")
print(pi.filter((pl.col("match_key") == k) & (pl.col("round") <= 3)).sort(["round", "player_id"]))
print("== player_info rows per round for match ==")
print(pi.filter(pl.col("match_key") == k).group_by("round").len().sort("round").head(20))

# how many rows per (match, round) in player_personal overall vs player_info
pp_cnt = pp.group_by(["match_key", "round"]).len()
pi_cnt = pi.group_by(["match_key", "round"]).len()
print("personal rows per (match,round): ", pp_cnt.group_by("len").len().sort("len").rows()[:15])
print("info rows per (match,round): ", pi_cnt.group_by("len").len().sort("len").rows()[:15])

# steam_id values: how many distinct per match
sid_pm = pp.group_by("match_key").agg(pl.col("steam_id").n_unique().alias("nsid"), pl.col("player_id").n_unique().alias("npid"))
print("distinct steam_id per match:", sid_pm.group_by("nsid").len().sort("nsid").rows())
print("distinct player_id per match:", sid_pm.group_by("npid").len().sort("npid").rows())
print("steam_id value counts:", pp.get_column("steam_id").value_counts().sort("count", descending=True).head(25).rows())
