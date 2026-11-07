"""D6-4 verifier, step 11: remaining claim items.
(a) win_reason side vs winner with the CORRECT mapping (T: 1,9,13,18; CT: 7,8,11,12,17), 2024 and round 13.
(b) round_end tick minus bomb tick (seconds at 64 ticks/s, ASSUMPTION T1) for affected vs control round 13.
(c) copy test: affected round_end13 record vs round_end12 record (winner, reason, tick).
(d) round_end13 tick inside round 13's own freeze window (fe13 .. fe14).
(e) 2026: which rounds carry the 106 score-vs-round_end disagreements.
Prints counts only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
TICK = 64  # ASSUMPTION T1
T_CODES = [1, 9, 13, 18]
CT_CODES = [7, 8, 11, 12, 17]

def tb(era, name):
    base = ROOT / "tomes" / ("2024-02-g" if era == "2024" else "2026-10-01_to_08") / "tome" / "csds"
    return [p for p in base.iterdir() if p.name.startswith(f"basic_{name}.")][0] / "dataframe_*"

re24 = pl.scan_parquet(str(tb("2024", "round_end"))).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
re26 = pl.scan_parquet(str(tb("2026", "round_end"))).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
for label, df in [("2024", re24), ("2026", re26)]:
    side = df.with_columns(pl.when(pl.col("win_reason_code").is_in(T_CODES)).then(2).when(pl.col("win_reason_code").is_in(CT_CODES)).then(3).otherwise(None).alias("reason_side"))
    ok = (side["reason_side"] == side["winner_team_code"]).sum()
    print(f"(a) {label}: reason side == winner in {ok} of {side.height} rows; unmapped={side['reason_side'].is_null().sum()}")
    r13 = side.filter(pl.col("round") == 13)
    print(f"(a) {label} round 13: reason side == winner in {(r13['reason_side']==r13['winner_team_code']).sum()} of {r13.height}")

x = pl.read_parquet(str(SCR / "verify_D6_4_physical_2024_r13.parquet"))
bs = pl.scan_parquet(str(tb("2024", "bomb_state"))).select(["match_key", "round", "tick", "event_type"]).collect()
bomb_tick = (bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"])
             .agg(pl.col("tick").min().alias("outcome_tick")))
rr = re24.filter(pl.col("round") == 13).join(bomb_tick.filter(pl.col("round") == 13), on=["match_key", "round"], how="inner")
rr = rr.join(x.select(["match_key", "round", "aff", "win"]), on=["match_key", "round"], how="inner")
rr = rr.with_columns(((pl.col("tick") - pl.col("outcome_tick")) / TICK).alias("re_minus_bomb_s"))
print("(b) round_end13 tick minus bomb outcome tick, seconds (all d01-25 rows with outcome):")
for lab, sub in [("affected d12-25", rr.filter(pl.col("aff") & (pl.col("win") == "d12-25"))),
                 ("control d12-25", rr.filter((~pl.col("aff")) & (pl.col("win") == "d12-25"))),
                 ("control d01-07", rr.filter((~pl.col("aff")) & (pl.col("win") == "d01-07")))]:
    v = sub["re_minus_bomb_s"]
    print(f"  {lab}: n={sub.height} median={v.median():.3f} min={v.min():.3f} max={v.max():.3f}; share exactly 429 ticks={(sub['tick']-sub['outcome_tick']==429).sum()}; share 0 ticks={(sub['tick']-sub['outcome_tick']==0).sum()}")

# (c) copy test: affected round_end13 vs round_end12 of the same match
r12 = re24.filter(pl.col("round") == 12).select(["match_key", pl.col("winner_team_code").alias("w12"), pl.col("win_reason_code").alias("r12"), pl.col("tick").alias("t12")])
r13 = re24.filter(pl.col("round") == 13).select(["match_key", "winner_team_code", "win_reason_code", "tick"]).join(r12, on="match_key", how="left")
aff_keys = x.filter(pl.col("aff") & (pl.col("win") == "d12-25")).select("match_key")
a = r13.join(aff_keys, on="match_key", how="inner")
print(f"(c) affected matches with round_end12 present: {a.filter(pl.col('w12').is_not_null()).height} of {a.height}")
print(f"(c) affected: round_end13 winner == round_end12 winner: {(a['winner_team_code']==a['w12']).sum()}; same reason code: {(a['win_reason_code']==a['r12']).sum()}; same tick: {(a['tick']==a['t12']).sum()}")

# (d) round_end13 tick inside round 13 freeze window
fe = (pl.scan_parquet(str(tb("2024", "round_state"))).select(["match_key", "round", "tick", "event_type"])
      .filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]).collect()
      .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("fe"))).sort(["match_key", "round"])
fe_n = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"fe": "fe_next"})
w = (r13.select(["match_key", "tick"]).join(fe.filter(pl.col("round") == 13).select(["match_key", "fe"]), on="match_key", how="left")
       .join(fe_n.filter(pl.col("round") == 13).select(["match_key", "fe_next"]), on="match_key", how="left"))
w = w.with_columns(((pl.col("tick") >= pl.col("fe")) & (pl.col("tick") < pl.col("fe_next"))).alias("in_window"))
wa = w.join(aff_keys, on="match_key", how="inner")
print(f"(d) affected round_end13 tick inside round-13 freeze window: {wa['in_window'].sum()} of {wa.height}")

# (e) 2026 score vs round_end disagreements: which rounds
j26 = pl.read_parquet(str(SCR / "verify_D6_4_join_2026.parquet")).filter(pl.col("score_side").is_not_null())
dis = j26.filter(pl.col("score_side") != pl.col("winner_team_code"))
print(f"(e) 2026 score-vs-round_end disagreements: {dis.height}; by round: {dis.group_by('round').len().sort('round').rows()}")
print(f"(e) 2026 agree excluding rounds 27/33/39/45: {j26.filter(~pl.col('round').is_in([27,33,39,45])).filter(pl.col('score_side')==pl.col('winner_team_code')).height}, disagree={j26.filter(~pl.col('round').is_in([27,33,39,45])).filter(pl.col('score_side')!=pl.col('winner_team_code')).height}")
