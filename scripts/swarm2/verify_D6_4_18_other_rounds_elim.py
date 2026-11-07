"""D6-4 verifier, step 18: elimination test for the other rounds with score-vs-round_end disagreement
(rounds 9, 25, 28; round 13 repeated for reference). 2024, player_count 20, days 12-25.
For rows where round_end and score disagree, count how often the loser (per each source) is fully wiped in the
death data. Round 27 is excluded (score check fails in 100% of rows; OT boundary artifact per A6-8).
Side codes T=2, CT=3 (ASSUMPTION P1). Counts only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
def ch(name):
    return [p for p in base.iterdir() if p.name.startswith(f"basic_{name}.")][0] / "dataframe_*"
spawn = pl.scan_parquet(str(ch("player_spawn"))).select(["match_key", "round", "player_id", "player_team_code"]).collect()
death = pl.scan_parquet(str(ch("player_death"))).select(["match_key", "round", "player_id", "player_team_code"]).collect()
roster = (spawn.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").mode().first().alias("side"))
          .group_by(["match_key", "round", "side"]).agg(pl.col("player_id").n_unique().alias("roster")))
dead = (death.group_by(["match_key", "round", "player_team_code"]).agg(pl.col("player_id").n_unique().alias("dead"))
        .rename({"player_team_code": "side"}))
st = roster.join(dead, on=["match_key", "round", "side"], how="left").with_columns(pl.col("dead").fill_null(0)).with_columns(
    (pl.col("dead") >= pl.col("roster")).alias("wiped"))
wT = st.filter(pl.col("side") == 2).select(["match_key", "round", pl.col("wiped").alias("T_wiped")])
wC = st.filter(pl.col("side") == 3).select(["match_key", "round", pl.col("wiped").alias("CT_wiped")])
j = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet")).filter(pl.col("player_count") == 20)
j = j.join(wT, on=["match_key", "round"], how="left").join(wC, on=["match_key", "round"], how="left")
j = j.filter(pl.col("day").is_in([f"{x:02d}" for x in range(12, 26)]) & pl.col("score_side").is_not_null())
def loser_wiped(sub, winner_col):
    loser_T = sub[winner_col] == 3
    return int(((loser_T & sub["T_wiped"].fill_null(False)) | ((~loser_T) & sub["CT_wiped"].fill_null(False))).sum())
for r in [9, 13, 25, 28]:
    s = j.filter(pl.col("round") == r)
    dis = s.filter(pl.col("winner_team_code") != pl.col("score_side"))
    print(f"round {r}: score-known={s.height} disagree={dis.height} | disagreeing rows: loser wiped per round_end winner={loser_wiped(dis, 'winner_team_code')}, per score winner={loser_wiped(dis, 'score_side')}")
    ok_nb = dis.filter(pl.col("bomb_side").is_null())
    if ok_nb.height:
        print(f"    of which no-bomb: n={ok_nb.height}, wiped per round_end={loser_wiped(ok_nb, 'winner_team_code')}, per score={loser_wiped(ok_nb, 'score_side')}")
