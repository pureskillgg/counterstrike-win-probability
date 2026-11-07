import glob, os
import polars as pl
cands = sorted(glob.glob("E:/csds_final/csds/2026/09/*/*/player_info"))
sample = cands[::max(1, len(cands) // 40)][:40]
tot_deaths = 0; null_deaths = 0; first_null = 0; matches_with_null = 0
for pif in sample:
    d = os.path.dirname(pif)
    pdth = pl.read_parquet(os.path.join(d, "player_death"), columns=["round", "tick", "player_id", "player_team_code"])
    tot_deaths += pdth.height
    nd = pdth.filter(pl.col("player_team_code").is_null())
    null_deaths += nd.height
    if nd.height:
        matches_with_null += 1
    # first death of that player in the round (affects alive counts)
    first = pdth.sort("tick").group_by(["round", "player_id"]).first()
    first_null += first.filter(pl.col("player_team_code").is_null()).height
print("player_death rows in sample", tot_deaths, "| null player_team_code", null_deaths, "| matches with any null", matches_with_null, "| first-death rows with null team", first_null)
