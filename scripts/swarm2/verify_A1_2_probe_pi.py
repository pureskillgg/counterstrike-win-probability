# verify A1-2: player_info and player_death detail for match m_36b24642
import glob, os
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
MK = "csds/2026/10/01/m_36b246426c63e2e81c3165c3/csds"
def scan(ch, cols):
    pages = sorted(glob.glob(os.path.join(T, f"basic_{ch}.2026-09-18,2026-10-08", "dataframe_*")))
    cols = list(dict.fromkeys(cols + ["match_key"]))
    return pl.concat([pl.scan_parquet(p).select(cols) for p in pages], how="diagonal_relaxed").filter(pl.col("match_key") == MK).collect()
pi = scan("player_info", ["round", "player_id", "team_code", "wins", "player_id_fixed"]).sort(["round", "player_id"])
print("player_info rows", pi.height)
print(pi.head(60).rows())
pd_ = scan("player_death", ["round", "tick", "player_id", "attacker_id", "player_team_code", "attacker_team_code", "assister_team_code"]).sort("tick")
print("player_death rows", pd_.height)
print(pd_.select(["round", "tick", "player_id", "player_team_code", "attacker_id", "attacker_team_code"]).head(80).rows())
