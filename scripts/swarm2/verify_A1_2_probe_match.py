# verify A1-2: raw look at one disputed match (7368f3ae) across player_info, round_end, score_update
import glob, os, sys
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
MK = "csds/2026/10/01/7368f3ae-7353-4f68-8d04-7c97f22ea2b8/csds"
def scan(ch, cols):
    pages = sorted(glob.glob(os.path.join(T, f"basic_{ch}.2026-09-18,2026-10-08", "dataframe_*")))
    return pl.concat([pl.scan_parquet(p).select(cols) for p in pages], how="diagonal_relaxed").filter(pl.col("match_key") == MK).collect()
pi = scan("player_info", ["round", "player_id", "team_code", "match_key"])
print("player_info rows", pi.height, "rounds", sorted(pi.get_column("round").unique().to_list()))
print(pi.group_by(["round", "team_code"]).len().sort(["round", "team_code"]).pivot(on="team_code", index="round", values="len").sort("round"))
re_ = scan("round_end", ["round", "tick", "winner_team_code", "match_key"]).sort("round")
print("round_end:", re_.select(["round", "tick", "winner_team_code"]).rows())
su = scan("score_update", ["round", "tick", "team_code", "old_score", "new_score", "match_key"]).sort(["tick"])
print("score_update rows", su.height)
for r in su.select(["round", "tick", "team_code", "old_score", "new_score"]).rows():
    print("  ", r)
ti = scan("team_change", ["round", "tick", "player_id", "old_team_code", "new_team_code", "match_key"]).sort("tick")
print("team_change rows", ti.height)
print(ti.select(["round", "tick", "player_id", "old_team_code", "new_team_code"]).head(40).rows())
