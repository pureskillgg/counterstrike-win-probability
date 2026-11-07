"""verify_D1_5_b: follow-ups to verify_D1_5_a.py for the D1-5 verification (2024 tome, read-only).

Question: is the D4 death-based agreement (winner_team_code agrees with the last victim side, 98.8% at round 13)
independent of winner_team_code, or does it share the same side mapping?

  E1 side of each death victim in player_death (player_team_code) vs the player_info roster side of the same player
  E2 alive-count elimination winner computed two ways: victim side from player_death vs from player_info
  E3 round 13 (d12-25): alive-count winner (player_info) vs winner_team_code, cross-tab
  E4 planter side at planted rounds: bomb_state code vs player_info vs player_death (if the planter died)
  E5 coverage: round-13 non-plant rows by category (decided / both alive / none / roster not 5v5) and their disagreement

Appends to logs\\swarm2\\verify_D1_5.log (written first by verify_D1_5_a.py).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
LOG = ROOT / "logs" / "swarm2" / "verify_D1_5.log"
_fh = open(LOG, "a", encoding="utf-8")
HI = [f"{d:02d}" for d in range(12, 26)]
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)


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


def dg_expr(col="match_key"):
    return (pl.when(pl.col(col).str.extract(r"csds/2024/02/(\d\d)/", 1).is_in(HI)).then(pl.lit("d12-25"))
            .otherwise(pl.lit("d01-07")).alias("dg"))


hdr("E0. load (2024 tome)")
RE = (rd("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
      .unique(["match_key", "round"], keep="first").rename({"tick": "end_tick"}))
RE20 = RE.filter(pl.col("player_count") == 20).with_columns(dg_expr())
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_id_fixed", "player_team_code"])
PLANT = (BS.filter(pl.col("event_type") == "bomb_planted")
         .group_by(["match_key", "round"])
         .agg(pl.col("tick").min().alias("plant_tick"),
              pl.col("player_id_fixed").first().cast(pl.Int64).alias("planter_pid"),
              pl.col("player_team_code").first().alias("planter_bs_code")))
EXP = BS.filter(pl.col("event_type") == "bomb_exploded").group_by(["match_key", "round"]).agg(pl.len().alias("n_exp"))
DEF = BS.filter(pl.col("event_type") == "bomb_defused").group_by(["match_key", "round"]).agg(pl.len().alias("n_def"))
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
DE = rd("player_death", ["match_key", "round", "tick", "player_id_fixed", "player_team_code"]).with_columns(
    pl.col("player_id_fixed").cast(pl.Int64).alias("pid"), pl.col("player_team_code").cast(pl.Int64).alias("dteam"))
emit("player_death rows:", DE.height, " player_info rows (after V7 exclusion):", PI.height,
     " excluded matches:", dupk.height)

# ------------------------------------------------------------------ E1
hdr("E1. victim side in player_death vs player_info roster side (same player, same round)")
VD = (DE.join(PI.select(["match_key", "round", "pid", "team_code"]), on=["match_key", "round", "pid"], how="inner")
        .with_columns((pl.col("dteam") == pl.col("team_code")).alias("same"), dg_expr()))
emit("death rows matched to roster:", VD.height, " of", DE.height)
emit("share of matched death rows where death team == roster team, by day group and round (rounds 1-30):")
e1 = (VD.filter(pl.col("round") <= 30).group_by(["dg", "round"])
        .agg(pl.len().alias("n"), (~pl.col("same")).sum().alias("death_ne_roster"),
             (~pl.col("same")).mean().round(4).alias("rate"))
        .sort(["dg", "round"]))
emit(e1.filter(pl.col("death_ne_roster") > 0))
emit("round 13 by day group:")
emit(e1.filter(pl.col("round") == 13))

# ------------------------------------------------------------------ roster, planted, and alive tests
ROS = PI.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("nT"),
                                              (pl.col("team_code") == 3).sum().alias("nCT"))
VIC_info = (DE.join(RE20.select(["match_key", "round", "end_tick"]), on=["match_key", "round"], how="inner")
              .filter(pl.col("tick") <= pl.col("end_tick"))
              .join(PI.select(["match_key", "round", "pid", "team_code"]), on=["match_key", "round", "pid"], how="inner")
              .unique(["match_key", "round", "pid"], keep="first")
              .group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("vT"),
                                                    (pl.col("team_code") == 3).sum().alias("vCT")))
VIC_death = (DE.join(RE20.select(["match_key", "round", "end_tick"]), on=["match_key", "round"], how="inner")
               .filter(pl.col("tick") <= pl.col("end_tick"))
               .unique(["match_key", "round", "pid"], keep="first")
               .group_by(["match_key", "round"]).agg((pl.col("dteam") == 2).sum().alias("vT"),
                                                     (pl.col("dteam") == 3).sum().alias("vCT")))


def alive_frame(vic):
    return (RE20.select(["match_key", "round", "dg", "winner_team_code", "end_tick"])
            .join(ROS, on=["match_key", "round"], how="left")
            .join(vic, on=["match_key", "round"], how="left")
            .join(PLANT.select(["match_key", "round", "planter_pid", "planter_bs_code"]), on=["match_key", "round"], how="left")
            .join(SC, on=["match_key", "round"], how="left")
            .with_columns(pl.col("vT").fill_null(0), pl.col("vCT").fill_null(0))
            .with_columns((pl.col("nT") - pl.col("vT")).alias("aT"), (pl.col("nCT") - pl.col("vCT")).alias("aCT"))
            .with_columns(
                pl.when((pl.col("aT") == 0) & (pl.col("aCT") > 0)).then(pl.lit(3))
                  .when((pl.col("aCT") == 0) & (pl.col("aT") > 0)).then(pl.lit(2))
                  .when((pl.col("aT") > 0) & (pl.col("aCT") > 0)).then(pl.lit(-1))
                  .otherwise(pl.lit(0)).cast(pl.Int64).alias("alive_win")))


AI = alive_frame(VIC_info)
AD = alive_frame(VIC_death)
planted_keys = PLANT.select(["match_key", "round"]).with_columns(pl.lit(True).alias("planted"))
AI = AI.join(planted_keys, on=["match_key", "round"], how="left").with_columns(pl.col("planted").fill_null(False))
AD = AD.join(planted_keys, on=["match_key", "round"], how="left").with_columns(pl.col("planted").fill_null(False))
five = (pl.col("nT") == 5) & (pl.col("nCT") == 5)

# ------------------------------------------------------------------ E2
hdr("E2. alive-count winner with victim side from player_death vs from player_info (non-plant, 5v5)")
NI = AI.filter(five & ~pl.col("planted") & pl.col("alive_win").is_in([2, 3]))
ND = AD.filter(five & ~pl.col("planted") & pl.col("alive_win").is_in([2, 3]))
J2 = (NI.select(["match_key", "round", "dg", "winner_team_code", "alive_win", "score_side"])
        .join(AD.select(["match_key", "round", pl.col("alive_win").alias("alive_win_death")]),
              on=["match_key", "round"], how="left"))
emit("decided non-plant 5v5 rounds (player_info sides):", J2.height)
emit("agreement by round (dg, round, n, winner!=alive_info, winner!=alive_death, alive_info!=alive_death, score!=alive_info):")
e2 = (J2.filter(pl.col("round").is_in([9, 13, 25, 27, 28]))
        .group_by(["dg", "round"]).agg(
            pl.len().alias("n"),
            (pl.col("winner_team_code") != pl.col("alive_win")).sum().alias("win_ne_alive_info"),
            (pl.col("winner_team_code") != pl.col("alive_win_death")).sum().alias("win_ne_alive_death"),
            (pl.col("alive_win") != pl.col("alive_win_death")).sum().alias("info_ne_death"),
            (pl.col("score_side").is_not_null() & (pl.col("score_side") != pl.col("alive_win"))).sum().alias("score_ne_alive_info"))
        .sort(["dg", "round"]))
emit(e2)
emit("all decided rounds, round 13 only, cross-tab alive_info x alive_death (d12-25):")
emit(J2.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25"))
     .group_by(["alive_win", "alive_win_death"]).len().sort(["alive_win", "alive_win_death"]))

# ------------------------------------------------------------------ E3
hdr("E3. round 13, d12-25, non-plant 5v5, decided by player_info: winner_team_code vs alive_info winner")
E3 = J2.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25"))
emit(E3.group_by(["alive_win", "winner_team_code"]).len().sort(["alive_win", "winner_team_code"]))
emit("winner_team_code == 5 - alive_info winner for all disagreeing rows? (binary codes => inverted):",
     E3.filter(pl.col("winner_team_code") != pl.col("alive_win")).height, "disagreeing rows, of which inverted:",
     E3.filter((pl.col("winner_team_code") != pl.col("alive_win")) & (pl.col("winner_team_code") == 5 - pl.col("alive_win"))).height)
emit("score_side vs alive_info winner in the same rows:",
     E3.filter(pl.col("score_side").is_not_null()).height, "known; disagreements:",
     E3.filter(pl.col("score_side").is_not_null() & (pl.col("score_side") != pl.col("alive_win"))).height)

# ------------------------------------------------------------------ E4
hdr("E4. planter side at planted rounds: bomb_state code vs player_info vs player_death (round 13 and rest)")
pinfo = PI.select(["match_key", "round", "pid", "team_code"]).rename({"pid": "planter_pid", "team_code": "planter_info"})
ddeath = (DE.select(["match_key", "round", "pid", "dteam"]).unique(["match_key", "round", "pid"], keep="first")
          .rename({"pid": "planter_pid", "dteam": "planter_death"}))
P = (PLANT.join(pinfo, on=["match_key", "round", "planter_pid"], how="left")
          .join(ddeath, on=["match_key", "round", "planter_pid"], how="left")
          .join(RE20.select(["match_key", "round", "dg", "winner_team_code"]), on=["match_key", "round"], how="inner")
          .join(EXP, on=["match_key", "round"], how="left").join(DEF, on=["match_key", "round"], how="left")
          .with_columns(pl.when(pl.col("n_exp").is_not_null()).then(pl.lit("exploded"))
                          .when(pl.col("n_def").is_not_null()).then(pl.lit("defused"))
                          .otherwise(pl.lit("none")).alias("ev"),
                        pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rb")))
emit("planted full rounds:", P.height)
emit("planter code agreement (bomb_state vs player_info), by dg and round bin:")
emit(P.group_by(["dg", "rb"]).agg(pl.len().alias("n"),
                                   (pl.col("planter_bs_code") == pl.col("planter_info")).sum().alias("bs_eq_info"),
                                   pl.col("planter_info").is_null().sum().alias("info_missing"))
     .sort(["dg", "rb"]))
PD = P.filter(pl.col("planter_death").is_not_null())
emit("planters who died in the same round (death row present):", PD.height)
emit("round 13 d12-25 planters who died: (planter_bs_code, planter_info, planter_death, winner, ev)")
emit(PD.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25"))
     .group_by(["planter_bs_code", "planter_info", "planter_death", "winner_team_code", "ev"]).len()
     .sort(["planter_bs_code", "planter_info", "planter_death", "winner_team_code", "ev"]))
emit("all planted rounds where planter death code != bomb_state code, by dg and round bin:")
emit(PD.filter(pl.col("planter_death") != pl.col("planter_bs_code")).group_by(["dg", "rb"]).len().sort(["dg", "rb"]))

# ------------------------------------------------------------------ E5
hdr("E5. coverage: round-13 full rounds with score side, by category (d12-25 and d01-07)")
C = (RE20.filter(pl.col("round") == 13).select(["match_key", "round", "dg", "winner_team_code"])
       .join(SC, on=["match_key", "round"], how="left")
       .join(planted_keys, on=["match_key", "round"], how="left")
       .join(ROS, on=["match_key", "round"], how="left")
       .join(AI.select(["match_key", "round", "alive_win"]), on=["match_key", "round"], how="left")
       .with_columns(pl.col("planted").fill_null(False))
       .filter(pl.col("score_side").is_not_null())
       .with_columns(
           pl.when(pl.col("planted")).then(pl.lit("planted"))
             .when(pl.col("nT").is_null()).then(pl.lit("no roster"))
             .when(~((pl.col("nT") == 5) & (pl.col("nCT") == 5))).then(pl.lit("roster not 5v5"))
             .when(pl.col("alive_win").is_in([2, 3])).then(pl.lit("decided by alive"))
             .when(pl.col("alive_win") == -1).then(pl.lit("both alive"))
             .otherwise(pl.lit("none alive")).alias("cat"),
           (pl.col("score_side") != pl.col("winner_team_code")).alias("ne")))
emit(C.group_by(["dg", "cat"]).agg(pl.len().alias("n"), pl.col("ne").sum().alias("winner_ne_score"),
                                   pl.col("ne").mean().round(4).alias("rate"))
     .sort(["dg", "cat"]))
emit("roster sizes (nT, nCT) for round-13 non-plant rows, d12-25, excluding 5v5:")
emit(C.filter((pl.col("dg") == "d12-25") & ~pl.col("planted") & ~((pl.col("nT") == 5) & (pl.col("nCT") == 5)))
     .group_by(["nT", "nCT"]).len().sort("len", descending=True).head(10))
_fh.close()
print("appended", LOG)
