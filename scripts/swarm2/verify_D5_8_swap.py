"""Verifier for D5-8, second pass: which rounds are inverted, and does the header/panel scoreboard agree with round_end?

Checks:
  S1. bomb timing sanity: planted < outcome < round_end tick within the same round (is the bomb assigned to the right round?)
  S2. reason codes on regime-A round-13 rows where the bomb says the opposite of round_end (what does the round_end record say?)
  S3. freeze-end boundary classification NS (no side swap) vs SW (side swap) per boundary, regime A vs B,
      using the RECORDED winner (as D5_14 did) and the bomb-corrected winner on bomb rounds only.
  S4. header test in regime A complete matches: winner-based totals with flips at chosen rounds vs header final score.
  S5. cs_win_panel_match final score (round_state) vs header final score, all matches.
Writes nothing to the repo. Intermediate data stays in the scratchpad.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_8_swap.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(40)

B = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_8")
SUF = "2023-12-28,2024-02-25"
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]


def rd(ch, cols):
    fs = sorted(glob.glob(str(B / f"basic_{ch}.{SUF}" / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in fs], how="vertical_relaxed").collect()


R = pl.read_parquet(SCR / "rounds2024.parquet").rename({"winner_team_code": "winner"})
H = pl.read_parquet(SCR / "header2024.parquet")

# ---------- S1. bomb timing ----------
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type"])
RE_ticks = R.select(["match_key", "round", "tick"]).rename({"tick": "end_tick"})
bp = (BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("plant_tick")))
bo = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("out_tick"),
                                            pl.col("event_type").first().alias("out_ev")))
TM = bo.join(bp, on=["match_key", "round"], how="left").join(RE_ticks, on=["match_key", "round"], how="left")
TM = TM.with_columns(ordered=(pl.col("plant_tick") < pl.col("out_tick")) & (pl.col("out_tick") <= pl.col("end_tick")),
                     out_before_end=(pl.col("out_tick") <= pl.col("end_tick")))
TM = TM.join(R.select(["match_key", "round", "regA", "day"]), on=["match_key", "round"], how="left")
print("S1. bomb timing: outcome tick within [plant, round_end] for the same round (by regime, late vs early):")
print(TM.with_columns(g=pl.when(pl.col("day").is_in(LATE)).then(pl.lit("late")).otherwise(pl.lit("early")))
      .group_by(["g", "regA", "ordered"]).len().sort(["g", "regA", "ordered"]))
print("   round-13 only, regime A, late:")
print(TM.filter((pl.col("round") == 13) & pl.col("regA") & pl.col("day").is_in(LATE)).group_by(["out_ev", "ordered"]).len())

# ---------- S2. reason codes and bomb-event team codes ----------
aff = R.filter((pl.col("round") == 13) & pl.col("regA") & pl.col("bomb_side").is_not_null()
               & (pl.col("bomb_side") != pl.col("winner")))
print(f"\nS2. regime A round-13 rows where bomb != round_end winner: {aff.height}")
print(aff.group_by(["win_reason_code", "winner", "bomb_side"]).len().sort("len", descending=True))
# baseline: how do reason codes relate to bomb outcome on ordinary rounds (all days, bomb rounds)?
bomb_rows = R.filter(pl.col("bomb_side").is_not_null()).with_columns(
    bgrp=pl.when((pl.col("round") == 13) & pl.col("regA")).then(pl.lit("r13 regA"))
    .when(pl.col("round") == 13).then(pl.lit("r13 regB"))
    .when(pl.col("regA") & pl.col("round").is_in([25, 28])).then(pl.lit("r25/28 regA"))
    .otherwise(pl.lit("other")))
print("baseline reason code by bomb outcome (bomb rounds only):")
print(bomb_rows.group_by(["bgrp", "bomb_side", "win_reason_code"]).len().sort(["bgrp", "bomb_side", "win_reason_code"]))
# team code on the bomb outcome event (defuser side / exploder side), from bomb_state
BS2 = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
oe = (BS2.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("out_ev"),
                                            pl.col("player_team_code").first().alias("out_team")))
oe = oe.join(R.select(["match_key", "round", "regA", "winner", "bomb_side"]), on=["match_key", "round"], how="left")
oe = oe.with_columns(aff13=(pl.col("round") == 13) & pl.col("regA") & (pl.col("bomb_side") != pl.col("winner")))
print("\nS2b. outcome-event player_team_code by outcome label (aff = round-13 regime-A rows where bomb != winner):")
print(oe.group_by(["aff13", "out_ev", "out_team"]).len().sort(["aff13", "out_ev", "out_team"]))

# ---------- S3. boundary classification per regime ----------
# uses fe(r), fe(r+1), and the recorded winner of round r; needs both rounds present in round_end
FE = R.select(["match_key", "round", "t_fe", "ct_fe", "winner", "regA", "day"])
N = FE.with_columns(nt=pl.col("t_fe").shift(-1).over("match_key"), nct=pl.col("ct_fe").shift(-1).over("match_key"),
                    nr=pl.col("round").shift(-1).over("match_key"))
N = N.filter((pl.col("nr") == pl.col("round") + 1) & pl.col("nt").is_not_null() & pl.col("winner").is_not_null())
a = pl.when(pl.col("winner") == 2).then(1).otherwise(0)
b = pl.when(pl.col("winner") == 3).then(1).otherwise(0)
N = N.with_columns(ns=(pl.col("nt") == pl.col("t_fe") + a) & (pl.col("nct") == pl.col("ct_fe") + b),
                   sw=(pl.col("nt") == pl.col("ct_fe") + b) & (pl.col("nct") == pl.col("t_fe") + a))
N = N.with_columns(cls=pl.when(pl.col("ns") & pl.col("sw")).then(pl.lit("both"))
                   .when(pl.col("ns")).then(pl.lit("NS"))
                   .when(pl.col("sw")).then(pl.lit("SW")).otherwise(pl.lit("neither")))
print("\nS3. boundary r->r+1 classification using recorded winner r, late days only, by regime:")
Nl = N.filter(pl.col("day").is_in(LATE)).filter(pl.col("round").is_in([8, 9, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29]))
print(Nl.group_by(["regA", "round", "cls"]).len().sort(["regA", "round", "cls"]))

# ---------- S4. header test on regime-A complete matches, with flips ----------
def side_x(r):
    return pl.when((pl.col("round") <= 12) | ((pl.col("round") >= 25) & (((pl.col("round") - 25) % 6) < 3))).then(pl.lit(2)).otherwise(pl.lit(3))


P = R.with_columns(sideX=side_x("round"))
cnt = P.group_by("match_key").agg(pl.len().alias("n_rounds"), pl.col("regA").first(),
                                  pl.col("t_starters_score_final").first().alias("hdr_t"),
                                  pl.col("ct_starters_score_final").first().alias("hdr_ct"),
                                  pl.col("day").first())
C = cnt.filter(pl.col("regA") & (pl.col("hdr_t") + pl.col("hdr_ct") == pl.col("n_rounds")))
PC = P.join(C.select(["match_key", "hdr_t", "hdr_ct", "n_rounds"]), on="match_key")
print(f"\nS4. regime A complete matches: {C.height}; header test by flip set (winner recorded, flipped at listed rounds):")
FLIPS = {"none": [], "13": [13], "25,28": [25, 28], "13,25,28": [13, 25, 28], "13,25,27,28": [13, 25, 27, 28]}
for name, rounds in FLIPS.items():
    w = pl.when(pl.col("round").is_in(rounds)).then(5 - pl.col("winner")).otherwise(pl.col("winner")) if rounds else pl.col("winner")
    Q = PC.with_columns(wf=w, xw=((pl.col("sideX") == w).cast(pl.Int64)))
    tot = Q.group_by("match_key").agg(pl.col("xw").sum().alias("X"), pl.col("hdr_t").first().alias("hdr_t"))
    ok = (tot["X"] == tot["hdr_t"]).sum()
    print(f"   flip {name:12s}: exact {ok:5d} / {tot.height}  ({100.0 * ok / tot.height:.2f}%)")

# which matches fail under the no-flip recorded winner, and do they fail on rounds 13/25/28 only?
Q0 = PC.with_columns(xw=((pl.col("sideX") == pl.col("winner")).cast(pl.Int64)))
tot0 = Q0.group_by("match_key").agg(pl.col("xw").sum().alias("X"), pl.col("hdr_t").first().alias("hdr_t"),
                                   pl.col("n_rounds").first())
print("   no-flip: signed difference (X - header) distribution:")
print(tot0.with_columns(d=pl.col("X") - pl.col("hdr_t")).group_by("d").len().sort("len", descending=True).head(10))

# ---------- S5. cs_win_panel_match vs header ----------
RS = rd("round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
pan = (RS.filter(pl.col("event_type") == "cs_win_panel_match").group_by("match_key")
       .agg(pl.col("t_score").max().alias("pan_t"), pl.col("ct_score").max().alias("pan_ct"),
            pl.col("t_score").min().alias("pan_t_min"), pl.col("ct_score").min().alias("pan_ct_min")))
PH = H.join(pan, on="match_key", how="left")
print("\nS5. cs_win_panel_match scores vs header final (all matches):")
print(PH.with_columns(eq_t=(pl.col("pan_t") == pl.col("t_starters_score_final")),
                      eq_ct=(pl.col("pan_ct") == pl.col("ct_starters_score_final")),
                      eq_swap=(pl.col("pan_t") == pl.col("ct_starters_score_final")))
      .group_by(["eq_t", "eq_ct", "eq_swap"]).len().sort("len", descending=True))
