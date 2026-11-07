"""verify_D1_5_a: independent verifier for claim D1-5 (2024 round-13 winner_team_code is the inverted label).

Reads only the 2024 tome G:\\pureskill\\csds-cs2\\tomes\\2024-02-g\\tome\\csds (read-only).
Writes only G:\\pureskill\\csds-cs2\\logs\\swarm2\\verify_D1_5.log.

Checks
  A  reproduce the claim's headline counts (bomb rounds, round-13 mismatches by day group)
  B  winner_team_code vs score side (round_state, claim rule A7) by round number, all full rounds
  C  physical elimination test on rounds WITHOUT a bomb plant. Alive players per side at round_end =
     player_info roster (5v5) minus player_death victims with tick <= round_end. A side with 0 alive lost.
     Independent of round_end winner, round_state scores, bomb events and the D4 3 s window.
  D  bomb timing: gap bomb_planted -> bomb_exploded / bomb_defused per group. A fixed C4 timer makes
     explosion gaps cluster; defuse gaps spread below the timer.

ASSUMPTIONS
  V1 full round = round_end.player_count == 20 (claim and D4 rule)
  V2 team_code 2 = T, 3 = CT (planter side is 2 in planted rounds)
  V3 score side = side whose score rose by exactly 1 between first and last round_state tick (A7)
  V4 roster = player_info rows of the round; full 5v5 required (nT == 5 and nCT == 5)
  V5 victim side = player_info team_code of the victim in that round
  V6 bomb timer = 40 s (competitive default); used to read the histogram, not to decide anything
  V7 matches with duplicate (match, round, player) rows in player_info excluded (3 matches, as D1)
  V8 no bomb planted and both sides alive at round_end = time-out (CT should win by rule)
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
LOG = ROOT / "logs" / "swarm2" / "verify_D1_5.log"
LOG.parent.mkdir(parents=True, exist_ok=True)
_fh = open(LOG, "w", encoding="utf-8")
HI_DAYS = [f"{d:02d}" for d in range(12, 26)]

pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)


def emit(*parts):
    s = " ".join(str(p) for p in parts)
    print(s, flush=True)
    _fh.write(s + "\n")
    _fh.flush()


def hdr(t):
    emit("\n" + "=" * 96 + "\n" + t + "\n" + "=" * 96)


def fdir(ch):
    d = sorted(BASE.glob(f"basic_{ch}.*"))
    assert len(d) == 1, (ch, d)
    return d[0]


def files(ch):
    return sorted(glob.glob(str(fdir(ch) / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


# ------------------------------------------------------------------ 0 schemas
hdr("0. schemas")
for ch in ["round_end", "bomb_state", "round_state", "player_info", "player_death"]:
    emit(ch, list(pl.read_parquet_schema(files(ch)[0]).keys()))

# ------------------------------------------------------------------ load
RE_all = rd("round_end", ["match_key", "round", "tick", "second", "winner_team_code", "player_count"])
conf = RE_all.group_by(["match_key", "round"]).agg(pl.len().alias("n"),
                                                   pl.col("winner_team_code").n_unique().alias("nw"))
hdr("0b. round_end keys")
emit("round_end rows:", RE_all.height, " (match,round) keys:", conf.height,
     " keys with >1 row:", conf.filter(pl.col("n") > 1).height,
     " keys with conflicting winner:", conf.filter(pl.col("nw") > 1).height)
RE = (RE_all.unique(["match_key", "round"], keep="first")
      .rename({"tick": "end_tick", "second": "end_sec"}))
RE20 = RE.filter(pl.col("player_count") == 20)
tick_ratio = RE.filter(pl.col("end_sec") > 0).select((pl.col("end_tick") / pl.col("end_sec")).median()).item()
emit("median round_end tick/second (tick rate estimate, not assumed):", round(tick_ratio, 3))

BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
emit("bomb_state event types:", BS.group_by("event_type").len().sort("event_type").rows())
PLANT = (BS.filter(pl.col("event_type") == "bomb_planted")
         .group_by(["match_key", "round"])
         .agg(pl.col("tick").min().alias("plant_tick"), pl.col("player_team_code").first().alias("planter_side")))
EXP = (BS.filter(pl.col("event_type") == "bomb_exploded")
       .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("exp_tick")))
DEF = (BS.filter(pl.col("event_type") == "bomb_defused")
       .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("def_tick")))
OUT = EXP.join(DEF, on=["match_key", "round"], how="full", coalesce=True)
emit("rounds: planted", PLANT.height, " exploded", EXP.height, " defused", DEF.height,
     " both exploded and defused", EXP.join(DEF, on=["match_key", "round"], how="inner").height)

RS = rd("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
SC = (RS.group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_b"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
        pl.col("t_score").sort_by("tick").last().alias("t_a"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_a")))
SC = SC.with_columns((pl.col("t_a") - pl.col("t_b")).alias("dT"),
                     (pl.col("ct_a") - pl.col("ct_b")).alias("dCT"))
SC = SC.with_columns(
    pl.when(pl.col("dT") == 1).then(pl.lit(2))
      .when(pl.col("dCT") == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).cast(pl.Int64).alias("score_side"))
emit("score_side counts over all rounds in round_state:", SC.group_by("score_side").len().sort("score_side").rows())

J = (RE20.join(OUT, on=["match_key", "round"], how="left")
         .join(PLANT, on=["match_key", "round"], how="left")
         .join(SC.select(["match_key", "round", "dT", "dCT", "score_side"]), on=["match_key", "round"], how="left")
         .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
         .with_columns(
            pl.when(pl.col("day").is_in(HI_DAYS)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"),
            pl.when(pl.col("round") == 13).then(pl.lit("r13"))
              .when(pl.col("round") == 25).then(pl.lit("r25"))
              .otherwise(pl.lit("other")).alias("rbin"),
            pl.when(pl.col("exp_tick").is_not_null()).then(pl.lit("exploded"))
              .when(pl.col("def_tick").is_not_null()).then(pl.lit("defused"))
              .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("ev"),
            pl.col("planter_side").is_not_null().alias("planted"))
         .with_columns(
            pl.when(pl.col("ev") == "exploded").then(pl.lit(2))
              .when(pl.col("ev") == "defused").then(pl.lit(3))
              .otherwise(pl.lit(None, dtype=pl.Int64)).cast(pl.Int64).alias("bomb_side"))
         .with_columns(
            (pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm_bomb"),
            (pl.col("score_side") == pl.col("winner_team_code")).alias("eq_score")))

# ------------------------------------------------------------------ A headline
hdr("A. reproduce claim headline numbers (full rounds, player_count == 20)")
B = J.filter(pl.col("ev").is_not_null())
emit("bomb-outcome full rounds:", B.height, " mismatches (bomb side != winner):", B.filter(pl.col("mm_bomb")).height,
     "  [claim/e2 log: 15436 / 499]")
emit(B.group_by(["rbin", "dg"]).agg(pl.len().alias("n"), pl.col("mm_bomb").sum().alias("mm"))
     .sort(["rbin", "dg"]))
R13B = B.filter(pl.col("round") == 13)
emit("round 13 bomb rows:", R13B.height, " score side == bomb side:",
     R13B.filter(pl.col("score_side") == pl.col("bomb_side")).height,
     " score side null:", R13B.filter(pl.col("score_side").is_null()).height)
emit("round 13 bomb rows, winner == score side (non-null):", R13B.filter(pl.col("eq_score")).height,
     " of", R13B.filter(pl.col("score_side").is_not_null()).height)

# ------------------------------------------------------------------ B winner vs score side
hdr("B. winner_team_code vs score side (round_state), all full rounds with non-null score side")
W = J.filter(pl.col("score_side").is_not_null())
emit("full rounds with score side:", W.height, " null score side:", J.height - W.height)
emit("round 13 by day group x bomb event (winner != score side counted):")
emit(W.filter(pl.col("round") == 13).group_by(["dg", "ev"], maintain_order=True)
     .agg(pl.len().alias("n"), (~pl.col("eq_score")).sum().alias("winner_ne_score")).sort(["dg", "ev"]))
RN = (W.group_by(["dg", "round"]).agg(pl.len().alias("n"), (~pl.col("eq_score")).sum().alias("dis"))
        .with_columns((pl.col("dis") / pl.col("n")).round(4).alias("rate")).sort(["dg", "round"]))
emit("winner != score side, rounds with any disagreement (dg, round, n, dis, rate):")
emit(RN.filter(pl.col("dis") > 0))

# ------------------------------------------------------------------ C alive test
hdr("C. alive-count elimination test on NON-PLANT full 5v5 rounds (independent of winner, scores, bomb)")
schema_de = pl.read_parquet_schema(files("player_death")[0])
pid_cands = [c for c in ["player_id_fixed", "player_id"] if c in schema_de]
PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"]).with_columns(
    pl.col("player_id_fixed").cast(pl.Int64).alias("pid"), pl.col("team_code").cast(pl.Int64))
dupk = (PI.group_by(["match_key", "round", "pid"]).len().filter(pl.col("len") > 1)
        .select("match_key").unique())
emit("player_info matches with duplicate (round, player) keys, excluded (V7):", dupk.height)
PI = PI.join(dupk, on="match_key", how="anti")
ROS = PI.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("nT"),
                                              (pl.col("team_code") == 3).sum().alias("nCT"))
DE_raw = rd("player_death", ["match_key", "round", "tick"] + pid_cands)
rost = PI.select(["match_key", "round", "pid"]).with_columns(pl.lit(1).alias("inr"))
best = None
for c in pid_cands:
    x = (DE_raw.select(["match_key", "round", "tick", pl.col(c).cast(pl.Int64).alias("pid")])
         .join(rost, on=["match_key", "round", "pid"], how="left"))
    rate = float(x["inr"].is_not_null().mean())
    emit(f"player_death id column {c}: share of death rows matching a roster player of the same round = {rate:.5f}")
    if best is None or rate > best[1]:
        best = (c, rate)
emit("using victim id column:", best[0])
DE = DE_raw.select(["match_key", "round", "tick", pl.col(best[0]).cast(pl.Int64).alias("pid")])
VIC = (DE.join(RE20.select(["match_key", "round", "end_tick"]), on=["match_key", "round"], how="inner")
         .filter(pl.col("tick") <= pl.col("end_tick"))
         .join(PI.select(["match_key", "round", "pid", "team_code"]), on=["match_key", "round", "pid"], how="inner")
         .unique(["match_key", "round", "pid"], keep="first")
         .group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("vdT"),
                                               (pl.col("team_code") == 3).sum().alias("vdCT")))
ALV = (RE20.select(["match_key", "round", "end_tick"])
       .join(ROS, on=["match_key", "round"], how="inner")
       .join(VIC, on=["match_key", "round"], how="left")
       .with_columns(pl.col("vdT").fill_null(0), pl.col("vdCT").fill_null(0))
       .with_columns((pl.col("nT") - pl.col("vdT")).alias("aT"), (pl.col("nCT") - pl.col("vdCT")).alias("aCT"))
       .filter((pl.col("nT") == 5) & (pl.col("nCT") == 5)))
A = (ALV.join(J.select(["match_key", "round", "winner_team_code", "score_side", "ev", "planted", "dg", "rbin"]),
              on=["match_key", "round"], how="inner")
        .with_columns(
            pl.when((pl.col("aT") == 0) & (pl.col("aCT") > 0)).then(pl.lit(3))
              .when((pl.col("aCT") == 0) & (pl.col("aT") > 0)).then(pl.lit(2))
              .when((pl.col("aT") > 0) & (pl.col("aCT") > 0)).then(pl.lit(-1))
              .otherwise(pl.lit(0)).cast(pl.Int64).alias("alive_win")))
emit("full 5v5 rounds with alive info:", A.height)
NP = A.filter(~pl.col("planted"))
emit("non-plant full 5v5 rounds:", NP.height, " planted rounds (5v5):", A.filter(pl.col("planted")).height,
     " of which with exploded/defused outcome:", A.filter(pl.col("planted") & pl.col("ev").is_not_null()).height)
emit("non-plant alive_win codes (2=T alive only, 3=CT alive only, -1=both alive, 0=none):")
emit(NP.group_by("alive_win").len().sort("alive_win"))

NPD = NP.with_columns(
    pl.col("score_side").is_not_null().alias("sk"),
    (pl.col("alive_win") == pl.col("winner_team_code")).alias("w_ok"),
    (pl.col("alive_win") == pl.col("score_side")).alias("s_ok"))
DEC = NPD.filter(pl.col("alive_win").is_in([2, 3]))
emit("DECIDED non-plant rounds (exactly one side alive at round_end):", DEC.height)
tabC = (DEC.group_by(["dg", "round"]).agg(
            pl.len().alias("n_dec"),
            (~pl.col("w_ok")).sum().alias("winner_wrong_vs_alive"),
            pl.col("sk").sum().alias("score_known"),
            (pl.col("sk") & ~pl.col("s_ok")).sum().alias("score_wrong_vs_alive"))
        .sort(["dg", "round"]))
emit("per round (decided rounds only): winner_wrong_vs_alive and score_wrong_vs_alive:")
emit(tabC.filter(pl.col("round").is_in([1, 9, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30])))
emit("summary by day group and round bin:")
emit(DEC.group_by(["dg", "rbin"]).agg(
        pl.len().alias("n_dec"),
        (~pl.col("w_ok")).sum().alias("winner_wrong_vs_alive"),
        pl.col("sk").sum().alias("score_known"),
        (pl.col("sk") & ~pl.col("s_ok")).sum().alias("score_wrong_vs_alive"))
     .sort(["dg", "rbin"]))

TO = NPD.filter(pl.col("alive_win") == -1)
hdr("C2. time-out candidates: non-plant rounds with both sides alive at round_end (CT should win by rule, V8)")
emit(TO.group_by(["dg", "rbin"]).agg(
        pl.len().alias("n"),
        (pl.col("winner_team_code") == 3).sum().alias("winner_CT"),
        (pl.col("winner_team_code") == 2).sum().alias("winner_T"),
        (pl.col("score_side") == 3).sum().alias("score_CT"),
        (pl.col("score_side") == 2).sum().alias("score_T"))
     .sort(["dg", "rbin"]))
emit("non-plant rounds with NO side alive (alive_win 0), round 13 by dg:")
emit(NPD.filter((pl.col("alive_win") == 0) & (pl.col("round") == 13)).group_by("dg").len())

# ------------------------------------------------------------------ D bomb timing
hdr("D. bomb timing: gap bomb_planted -> bomb outcome (ticks), by label, mismatch status and round bin")
TD = (J.filter(pl.col("ev").is_not_null() & pl.col("plant_tick").is_not_null())
        .with_columns(
            pl.when(pl.col("ev") == "exploded").then(pl.col("exp_tick")).otherwise(pl.col("def_tick")).alias("out_tick"))
        .with_columns(
            (pl.col("out_tick") - pl.col("plant_tick")).alias("gap_pe"),
            (pl.col("end_tick") - pl.col("out_tick")).alias("gap_oe"),
            pl.when(pl.col("mm_bomb")).then(pl.lit("MISMATCH")).otherwise(pl.lit("clean")).alias("st")))
emit("bomb rounds with plant tick:", TD.height)
TDG = (TD.group_by(["rbin", "st", "ev"]).agg(
        pl.len().alias("n"),
        pl.col("gap_pe").min().alias("min"),
        pl.col("gap_pe").quantile(0.05, "nearest").alias("p05"),
        pl.col("gap_pe").quantile(0.5, "nearest").alias("p50"),
        pl.col("gap_pe").quantile(0.95, "nearest").alias("p95"),
        pl.col("gap_pe").max().alias("max"),
        pl.col("gap_oe").median().alias("oe_p50"),
        pl.col("gap_oe").max().alias("oe_max"))
     .sort(["rbin", "st", "ev"]))
emit(TDG)
emit("tick gap for clean exploded rounds, 5 most common values (timer signature):")
emit(TD.filter((pl.col("st") == "clean") & (pl.col("ev") == "exploded"))
     .group_by("gap_pe").len().sort("len", descending=True).head(5))
emit("MISMATCH rounds, gap_pe histogram of 'exploded'-labelled rows (all bins, top 8 values):")
emit(TD.filter((pl.col("st") == "MISMATCH") & (pl.col("ev") == "exploded"))
     .group_by("gap_pe").len().sort("len", descending=True).head(8))
emit("MISMATCH rounds, gap_pe histogram of 'defused'-labelled rows (top 8 values):")
emit(TD.filter((pl.col("st") == "MISMATCH") & (pl.col("ev") == "defused"))
     .group_by("gap_pe").len().sort("len", descending=True).head(8))

# ------------------------------------------------------------------ summary
hdr("SUMMARY (numbers for the verdict)")
r13_dis = tabC.filter(pl.col("round") == 13)
emit("round 13 decided non-plant rounds (dg, n_dec, winner_wrong_vs_alive, score_wrong_vs_alive):")
emit(r13_dis)
emit("round 13 bomb rounds (dg, ev, mismatch):", B.filter(pl.col("round") == 13).group_by(["dg", "ev", "mm_bomb"])
     .len().sort(["dg", "ev", "mm_bomb"]).rows())
_fh.close()
print("wrote", LOG)
