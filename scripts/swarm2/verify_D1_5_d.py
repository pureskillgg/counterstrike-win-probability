"""verify_D1_5_d: scope of the round-winner inversion across ALL round numbers, both eras, plus coverage.

Per round number (2024 split by day group; 2026 as one group), using the same two physical references as _a and _c:
  - alive-count winner on NON-PLANT 5v5 rounds (roster from player_info, deaths from player_death, sides from player_info)
  - bomb outcome on planted rounds (bomb_state exploded/defused), where the bomb side is the reference
and the score side (round_state) as the claim's reference.
Also: coverage of planted rounds with a bomb outcome event, and whether bomb_defuse / bomb_action tables hold
additional outcome rounds. Round 9 bomb mismatches get a timer check.

ASSUMPTIONS as in verify_D1_5_a.py (V1 full round = player_count pc; V2 code 2 = T; V3 score rule A7; V4 5v5 roster;
V5 victim side from player_info; V6 40 s timer; V7 duplicate-key matches excluded in 2024 only).
Appends to logs\\swarm2\\verify_D1_5.log.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
ERAS = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", 20),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", 10),
}
LOG = ROOT / "logs" / "swarm2" / "verify_D1_5.log"
_fh = open(LOG, "a", encoding="utf-8")
HI = [f"{d:02d}" for d in range(12, 26)]
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)


def emit(*parts):
    s = " ".join(str(p) for p in parts)
    print(s, flush=True)
    _fh.write(s + "\n")
    _fh.flush()


def hdr(t):
    emit("\n" + "=" * 96 + "\n" + t + "\n" + "=" * 96)


def run_era(era, base, pc):
    def fdir(ch):
        d = sorted(base.glob(f"basic_{ch}.*"))
        assert len(d) == 1, (era, ch, d)
        return d[0]

    def rd(ch, cols):
        fs = sorted(glob.glob(str(fdir(ch) / "dataframe_*")))
        return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")

    if era == "2024":
        dg = (pl.when(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).is_in(HI))
              .then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
    else:
        dg = pl.lit("all").alias("dg")

    hdr(f"ERA {era} (full round player_count == {pc})")
    RE = (rd("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
          .unique(["match_key", "round"], keep="first").rename({"tick": "end_tick"}))
    RE20 = RE.filter(pl.col("player_count") == pc).with_columns(dg)
    BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
    PLANT = (BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"])
             .agg(pl.col("tick").min().alias("plant_tick"), pl.col("player_team_code").first().alias("planter_side")))
    EXP = (BS.filter(pl.col("event_type") == "bomb_exploded").group_by(["match_key", "round"])
           .agg(pl.col("tick").min().alias("exp_tick")))
    DEF = (BS.filter(pl.col("event_type") == "bomb_defused").group_by(["match_key", "round"])
           .agg(pl.col("tick").min().alias("def_tick")))
    RS = rd("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
    SC = (RS.group_by(["match_key", "round"]).agg(
            pl.col("t_score").sort_by("tick").first().alias("t_b"),
            pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
            pl.col("t_score").sort_by("tick").last().alias("t_a"),
            pl.col("ct_score").sort_by("tick").last().alias("ct_a")))
    SC = SC.with_columns(
        pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2))
          .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).cast(pl.Int64).alias("score_side")).select(
            ["match_key", "round", "score_side"])
    PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"]).with_columns(
        pl.col("player_id_fixed").cast(pl.Int64).alias("pid"), pl.col("team_code").cast(pl.Int64))
    dupk = PI.group_by(["match_key", "round", "pid"]).len().filter(pl.col("len") > 1).select("match_key").unique()
    PI = PI.join(dupk, on="match_key", how="anti")
    ROS = PI.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("nT"),
                                                  (pl.col("team_code") == 3).sum().alias("nCT"))
    DE = rd("player_death", ["match_key", "round", "tick", "player_id_fixed"]).with_columns(
        pl.col("player_id_fixed").cast(pl.Int64).alias("pid"))
    VIC = (DE.join(RE20.select(["match_key", "round", "end_tick"]), on=["match_key", "round"], how="inner")
             .filter(pl.col("tick") <= pl.col("end_tick"))
             .join(PI.select(["match_key", "round", "pid", "team_code"]), on=["match_key", "round", "pid"], how="inner")
             .unique(["match_key", "round", "pid"], keep="first")
             .group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("vT"),
                                                   (pl.col("team_code") == 3).sum().alias("vCT")))
    J = (RE20.join(PLANT, on=["match_key", "round"], how="left")
             .join(EXP, on=["match_key", "round"], how="left").join(DEF, on=["match_key", "round"], how="left")
             .join(SC, on=["match_key", "round"], how="left")
             .join(ROS, on=["match_key", "round"], how="left").join(VIC, on=["match_key", "round"], how="left"))
    J = J.with_columns(pl.col("vT").fill_null(0), pl.col("vCT").fill_null(0))
    J = J.with_columns(
        (pl.col("nT") - pl.col("vT")).alias("aT"), (pl.col("nCT") - pl.col("vCT")).alias("aCT"),
        pl.col("planter_side").is_not_null().alias("planted"),
        ((pl.col("nT") == 5) & (pl.col("nCT") == 5)).alias("five"),
        pl.when(pl.col("exp_tick").is_not_null()).then(pl.lit(2))
          .when(pl.col("def_tick").is_not_null()).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).cast(pl.Int64).alias("bomb_side"))
    J = J.with_columns(
        pl.when((pl.col("aT") == 0) & (pl.col("aCT") > 0)).then(pl.lit(3))
          .when((pl.col("aCT") == 0) & (pl.col("aT") > 0)).then(pl.lit(2))
          .when((pl.col("aT") > 0) & (pl.col("aCT") > 0)).then(pl.lit(-1))
          .otherwise(pl.lit(0)).cast(pl.Int64).alias("alive_win"))
    n_full = J.height
    n_planted = J.filter(pl.col("planted")).height
    n_planted_out = J.filter(pl.col("planted") & (pl.col("exp_tick").is_not_null() | pl.col("def_tick").is_not_null())).height
    emit(f"full rounds: {n_full}; planted full rounds: {n_planted}; with exploded/defused in bomb_state: {n_planted_out}; "
         f"planted without outcome: {n_planted - n_planted_out}")
    emit("bomb_state planted rows:", BS.filter(pl.col("event_type") == "bomb_planted").height,
         " exploded rows:", BS.filter(pl.col("event_type") == "bomb_exploded").height,
         " defused rows:", BS.filter(pl.col("event_type") == "bomb_defused").height)

    # coverage cross-check against the other bomb tables, if present
    for ch in ["bomb_defuse", "bomb_action"]:
        try:
            d = sorted(base.glob(f"basic_{ch}.*"))
            if not d:
                emit(ch, "table not present")
                continue
            fs = sorted(glob.glob(str(d[0] / "dataframe_*")))
            cols = list(pl.read_parquet_schema(fs[0]).keys())
            emit(ch, "columns:", cols)
            if "round" in cols and "match_key" in cols:
                t = rd(ch, ["match_key", "round"] + [c for c in ["event_type"] if c in cols])
                keys = t.select(["match_key", "round"]).unique()
                planted_keys = PLANT.select(["match_key", "round"])
                j = planted_keys.join(keys, on=["match_key", "round"], how="inner")
                emit(f"  distinct (match,round) in {ch}: {keys.height}; of planted rounds (all): {j.height}")
                pout = (J.filter(pl.col("planted") & pl.col("exp_tick").is_null() & pl.col("def_tick").is_null())
                        .select(["match_key", "round"]).join(keys, on=["match_key", "round"], how="inner"))
                emit(f"  planted rounds WITHOUT bomb_state outcome that have a row in {ch}: {pout.height}")
        except Exception as e:  # pragma: no cover - diagnostic only
            emit(ch, "could not be read:", repr(e)[:200])

    # per-round scope table
    J2 = J.filter(pl.col("score_side").is_not_null() | pl.col("bomb_side").is_not_null() | (pl.col("alive_win") != 0))
    J2 = J2.with_columns(
        (pl.col("winner_team_code") != pl.col("score_side")).alias("ne_score"),
        (pl.col("bomb_side") != pl.col("winner_team_code")).alias("ne_bomb"),
        (pl.col("alive_win") != pl.col("winner_team_code")).alias("ne_alive"))
    alive_ok = J2.filter(~pl.col("planted") & pl.col("five") & pl.col("alive_win").is_in([2, 3]))
    bomb_ok = J2.filter(pl.col("bomb_side").is_not_null())
    scope = (J2.with_columns(pl.lit(1).alias("one"))
             .group_by(["dg", "round"]).agg(
                 pl.col("one").sum().alias("n_full"),
                 (pl.col("five") & ~pl.col("planted") & pl.col("alive_win").is_in([2, 3])).sum().alias("n_alive_dec"),
                 (pl.col("five") & ~pl.col("planted") & pl.col("alive_win").is_in([2, 3]) & pl.col("ne_alive")).sum().alias("alive_winner_wrong"),
                 (pl.col("five") & ~pl.col("planted") & pl.col("alive_win").is_in([2, 3]) & pl.col("score_side").is_not_null()).sum().alias("alive_dec_score_known"),
                 (pl.col("five") & ~pl.col("planted") & pl.col("alive_win").is_in([2, 3]) & pl.col("score_side").is_not_null()
                  & (pl.col("score_side") != pl.col("alive_win"))).sum().alias("alive_dec_score_wrong"),
                 pl.col("bomb_side").is_not_null().sum().alias("n_bomb"),
                 (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).sum().alias("bomb_winner_wrong"),
                 (pl.col("bomb_side").is_not_null() & pl.col("score_side").is_not_null() & (pl.col("bomb_side") != pl.col("score_side"))).sum().alias("bomb_score_wrong"))
             .sort(["dg", "round"]))
    flag = scope.filter((pl.col("alive_winner_wrong") > 0) | (pl.col("alive_dec_score_wrong") > 0)
                        | (pl.col("bomb_winner_wrong") > 0) | (pl.col("bomb_score_wrong") > 0))
    emit("rounds with ANY winner/score/bomb disagreement (per round number, per day group):")
    emit(flag)
    emit("round 13, all rows (scope check):")
    emit(scope.filter(pl.col("round") == 13))
    emit("round numbers 1..40 with decided alive rows, per group, totals:")
    emit(scope.group_by("dg").agg(pl.col("n_alive_dec").sum().alias("alive_dec_total"),
                                  pl.col("alive_winner_wrong").sum().alias("alive_winner_wrong_total"),
                                  pl.col("alive_dec_score_wrong").sum().alias("alive_score_wrong_total"),
                                  pl.col("n_bomb").sum().alias("bomb_rows_total"),
                                  pl.col("bomb_winner_wrong").sum().alias("bomb_winner_wrong_total"),
                                  pl.col("bomb_score_wrong").sum().alias("bomb_score_wrong_total")))
    # round 9 bomb mismatch timing
    r9 = (J.filter((pl.col("round") == 9) & pl.col("planted") & (pl.col("bomb_side").is_not_null()) &
                   (pl.col("bomb_side") != pl.col("winner_team_code")))
          .with_columns(pl.when(pl.col("exp_tick").is_not_null()).then(pl.col("exp_tick")).otherwise(pl.col("def_tick")).alias("out_tick"))
          .join(PLANT.select(["match_key", "round", "plant_tick"]), on=["match_key", "round"], how="left")
          .with_columns((pl.col("out_tick") - pl.col("plant_tick")).alias("gap_pe")))
    emit("round 9 bomb mismatches: count", r9.height, " gap_pe (ticks) by label:")
    emit(r9.group_by(pl.when(pl.col("exp_tick").is_not_null()).then(pl.lit("exploded")).otherwise(pl.lit("defused")).alias("ev"))
         .agg(pl.len().alias("n"), pl.col("gap_pe").sort().alias("gaps")))
    return J


j24 = run_era("2024", *ERAS["2024"])
j26 = run_era("2026", *ERAS["2026"])

hdr("2026 control: round 13 alive / score / bomb agreement (claim: zero round-13 disagreements in 1,504 rounds)")
r13 = j26.filter(pl.col("round") == 13)
emit("2026 round 13 full rounds:", r13.height,
     " score known:", r13.filter(pl.col("score_side").is_not_null()).height,
     " winner != score:", r13.filter(pl.col("score_side").is_not_null() & (pl.col("score_side") != pl.col("winner_team_code"))).height,
     " alive-decided non-plant:", r13.filter(~pl.col("planted") & pl.col("five") & pl.col("alive_win").is_in([2, 3])).height,
     " winner != alive:", r13.filter(~pl.col("planted") & pl.col("five") & pl.col("alive_win").is_in([2, 3]) & (pl.col("alive_win") != pl.col("winner_team_code"))).height,
     " bomb rows:", r13.filter(pl.col("bomb_side").is_not_null()).height,
     " bomb != winner:", r13.filter(pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).height)
_fh.close()
print("appended", LOG)
