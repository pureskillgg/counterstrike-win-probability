"""D6-4 verifier, step 14: elimination test for round-13 rows WITHOUT a bomb outcome (days 12-25 and 01-07).

For each round: roster per side (distinct player_ids that spawned with that side code in the round) and the
number of distinct victims per side from player_death. A side is fully eliminated when all roster players died.
If a round ends by elimination, the losing side must be fully eliminated.
  - round_end winner R, and reason code in {8,9} (elimination) -> loser = other side of R: is it fully eliminated?
  - score winner S (freeze_end delta) -> loser = other side of S: is it fully eliminated?
Also: among no-bomb rows where round_end and score disagree, which one is consistent with the death data?
Side codes: T=2, CT=3 (ASSUMPTION P1 as in step 7; the planter code is 2 in this tome).
Prints counts only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
def ch(name):
    return [p for p in base.iterdir() if p.name.startswith(f"basic_{name}.")][0] / "dataframe_*"
spawn = pl.scan_parquet(str(ch("player_spawn"))).select(["match_key", "round", "player_id", "player_team_code"]).collect()
death = pl.scan_parquet(str(ch("player_death"))).select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
roster = (spawn.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").mode().first().alias("side"))
          .group_by(["match_key", "round", "side"]).agg(pl.col("player_id").n_unique().alias("roster")))
dead = death.group_by(["match_key", "round", "player_team_code"]).agg(pl.col("player_id").n_unique().alias("dead")).rename({"player_team_code": "side"})
side_state = roster.join(dead, on=["match_key", "round", "side"], how="left").with_columns(pl.col("dead").fill_null(0)).with_columns(
    (pl.col("dead") >= pl.col("roster")).alias("wiped"))
wiped_t = side_state.filter(pl.col("side") == 2).select(["match_key", "round", pl.col("wiped").alias("T_wiped"), pl.col("roster").alias("T_roster")])
wiped_ct = side_state.filter(pl.col("side") == 3).select(["match_key", "round", pl.col("wiped").alias("CT_wiped"), pl.col("roster").alias("CT_roster")])
j = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet")).filter(pl.col("player_count") == 20)
rs = pl.scan_parquet(str(ch("round_end"))).select(["match_key", "round", "win_reason_code"]).collect()
j = j.join(rs, on=["match_key", "round"], how="left").join(wiped_t, on=["match_key", "round"], how="left").join(wiped_ct, on=["match_key", "round"], how="left")
d = j.filter(pl.col("round") == 13).with_columns(
    pl.when(pl.col("day").is_in([f"{x:02d}" for x in range(12, 26)])).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("win"))
nb = d.filter(pl.col("bomb_side").is_null() & pl.col("score_side").is_not_null())
print("no-bomb round-13 rows with score side:", nb.group_by("win").len().rows())
for win in ["d01-07", "d12-25"]:
    s = nb.filter(pl.col("win") == win)
    dis = s.filter(pl.col("winner_team_code") != pl.col("score_side"))
    agr = s.filter(pl.col("winner_team_code") == pl.col("score_side"))
    print(f"\n[{win}] no-bomb: agree={agr.height} disagree={dis.height}")
    # reason codes
    print("  reason codes on disagreeing rows:", dis.group_by("win_reason_code").len().sort("win_reason_code").rows())
    # elimination consistency of round_end claim: loser per round_end must be wiped
    def loser_wiped(sub, winner_col):
        loser_is_T = sub[winner_col] == 3  # winner CT -> loser T
        t = sub["T_wiped"].fill_null(False)
        c = sub["CT_wiped"].fill_null(False)
        return ((loser_is_T & t) | ((~loser_is_T) & c)).sum(), sub.height
    ok_re, n_re = loser_wiped(s.filter(pl.col("win_reason_code").is_in([8, 9])), "winner_team_code")
    ok_sc, n_sc = loser_wiped(s.filter(pl.col("win_reason_code").is_in([8, 9])), "score_side")
    print(f"  elimination-coded rows (reason 8/9): n={n_re}; loser fully wiped per round_end winner: {ok_re}; per score winner: {ok_sc}")
    # on disagreeing rows: which claim is consistent with a wiped loser
    dr = dis.filter(pl.col("win_reason_code").is_in([8, 9]))
    ok_re_d, n_d = loser_wiped(dr, "winner_team_code")
    ok_sc_d, _ = loser_wiped(dr, "score_side")
    print(f"  disagreeing elimination-coded rows: n={n_d}; loser wiped per round_end: {ok_re_d}; per score: {ok_sc_d}")
    dd = dis.select(["T_wiped", "CT_wiped"]).group_by(["T_wiped", "CT_wiped"]).len().sort(["T_wiped", "CT_wiped"], nulls_last=True)
    print("  disagreeing rows (T_wiped, CT_wiped) counts:", dd.rows())
