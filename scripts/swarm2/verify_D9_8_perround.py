"""verify D9-8, part 4: per-round-number table of round_end disagreement with two independent sources,
by build (2024 tome, read-only). Also a sanity check of header final scores against round_state.

Sources:
  T1  score winner from the NEXT round's round_freeze_end pre-score minus this round's (no swap between;
      rounds 12, 24, 27, 30, ... excluded, ASSUMPTION A5 of verify_D9_8_scores.py). All rounds.
  E   elimination winner from player_death vs player_info roster, NON-bomb rounds with a generic
      message only (ASSUMPTIONS E1-E3 of verify_D9_8_elim.py). Contradiction = round_end winner
      disagrees with the side that was fully eliminated (determinable rounds only).
  H   header final score vs round_state final score (last round_end row), build 9842 and 9961 where
      round_state round_end exists (9842 only, per verify_D9_8_recompute.py S4 and S7).

Output is printed; no files are written under E:.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
GEN_T = "#SFUI_Notice_Terrorists_Win"
GEN_CT = "#SFUI_Notice_CTs_Win"
EXCL = [12, 24, 27, 30, 33, 36, 39, 42, 45, 48]


def files_of(name):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    return sorted(glob.glob(str(fd / "dataframe_*")))


def lz(name, cols):
    lf = pl.scan_parquet(files_of(name))
    have = lf.collect_schema().names()
    miss = [c for c in cols if c not in have]
    if miss:
        raise SystemExit(f"{name} missing columns {miss}")
    return lf.select(cols)


def sec(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


H = lz("header", ["key", "build_num", "t_starters_score_final", "ct_starters_score_final"]).collect()
H = H.rename({"key": "match_key"})
RE = lz("round_end", ["match_key", "round", "winner_team_code", "win_reason_message"]).collect()
RE = RE.unique(subset=["match_key", "round"], keep="first")
BO = lz("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).select(["match_key", "round"]).unique().collect()
BO = BO.with_columns(pl.lit(True).alias("has_bomb"))
FE = lz("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("event_type") == "round_freeze_end").collect()
FE = FE.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("t_score").first().alias("t_pre"), pl.col("ct_score").first().alias("ct_pre"))
F = FE.sort(["match_key", "round"]).with_columns(
    pl.col("round").shift(-1).over("match_key").alias("next_round"),
    pl.col("t_pre").shift(-1).over("match_key").alias("t_next"),
    pl.col("ct_pre").shift(-1).over("match_key").alias("ct_next"),
).filter(pl.col("next_round") == pl.col("round") + 1).with_columns(
    (pl.col("t_next") - pl.col("t_pre")).alias("dT"),
    (pl.col("ct_next") - pl.col("ct_pre")).alias("dCT"),
).with_columns(
    pl.when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
      .when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("t1_winner"),
).filter(~pl.col("round").is_in(EXCL)).select(["match_key", "round", "t1_winner"])

PI = lz("player_info", ["match_key", "round", "player_id", "team_code"]).collect()
DE = lz("player_death", ["match_key", "round", "player_id", "player_team_code"]).collect()
roster = (PI.filter(pl.col("team_code").is_in([2, 3])).unique(subset=["match_key", "round", "player_id"])
          .group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("nT"),
                                                 (pl.col("team_code") == 3).sum().alias("nCT")))
victims = (DE.filter(pl.col("player_team_code").is_in([2, 3]))
           .unique(subset=["match_key", "round", "player_id", "player_team_code"])
           .group_by(["match_key", "round"]).agg((pl.col("player_team_code") == 2).sum().alias("dT"),
                                                  (pl.col("player_team_code") == 3).sum().alias("dCT")))

J = (RE.join(H.select(["match_key", "build_num"]), on="match_key", how="left")
       .join(BO, on=["match_key", "round"], how="left")
       .join(F, on=["match_key", "round"], how="left")
       .join(roster, on=["match_key", "round"], how="left")
       .join(victims, on=["match_key", "round"], how="left")
       .with_columns(pl.col("has_bomb").fill_null(False), pl.col("dT").fill_null(0), pl.col("dCT").fill_null(0)))
J = J.with_columns(
    (pl.col("winner_team_code") != pl.col("t1_winner")).alias("t1_disagree"),
    pl.when(pl.col("nT").is_not_null() & pl.col("nCT").is_not_null() & (pl.col("nT") > 0) & (pl.col("nCT") > 0)
            & (~pl.col("has_bomb")) & pl.col("win_reason_message").is_in([GEN_T, GEN_CT])
            & (((pl.col("dT") >= pl.col("nT")) & ~(pl.col("dCT") >= pl.col("nCT")))
               | ((pl.col("dCT") >= pl.col("nCT")) & ~(pl.col("dT") >= pl.col("nT")))))
      .then(pl.when((pl.col("dT") >= pl.col("nT")) & ~(pl.col("dCT") >= pl.col("nCT"))).then(pl.lit(3))
            .otherwise(pl.lit(2)))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_winner"),
)
J = J.with_columns((pl.col("winner_team_code") != pl.col("elim_winner")).alias("elim_contra"))

sec("P1 per round number, build 9961: T1 disagreement and elimination contradiction (rounds 1-30)")
T = J.filter(pl.col("build_num") == 9961)
tab = T.group_by("round").agg(
    pl.col("t1_winner").is_not_null().sum().alias("n_t1"),
    pl.col("t1_disagree").sum().alias("t1_disagree"),
    pl.col("elim_winner").is_not_null().sum().alias("n_elim_det"),
    pl.col("elim_contra").sum().alias("elim_contra"),
    pl.col("has_bomb").sum().alias("n_bomb"),
).sort("round").filter(pl.col("round") <= 30)
tab = tab.with_columns(
    (pl.col("t1_disagree") / pl.col("n_t1")).round(3).alias("t1_rate"),
    (pl.col("elim_contra") / pl.col("n_elim_det")).round(3).alias("elim_rate"),
)
print(tab)

sec("P2 per round number, build 9842: same columns (expect ~0 disagreement)")
T8 = J.filter(pl.col("build_num") == 9842)
tab8 = T8.group_by("round").agg(
    pl.col("t1_winner").is_not_null().sum().alias("n_t1"),
    pl.col("t1_disagree").sum().alias("t1_disagree"),
    pl.col("elim_winner").is_not_null().sum().alias("n_elim_det"),
    pl.col("elim_contra").sum().alias("elim_contra"),
).sort("round").filter(pl.col("round") <= 30)
print(tab8)

sec("P3 totals: bomb-round vs non-bomb-round T1 disagreement, build 9961 round 13 and 25")
x = J.filter((pl.col("build_num") == 9961) & pl.col("round").is_in([13, 25]) & pl.col("t1_winner").is_not_null())
print(x.group_by(["round", "has_bomb"]).agg(pl.len().alias("n"), pl.col("t1_disagree").sum().alias("t1_disagree"),
                                            pl.col("t1_disagree").mean().alias("rate")).sort(["round", "has_bomb"]))

sec("H header final score vs round_state final score (build 9842 where round_state round_end exists)")
RSE = lz("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("event_type") == "round_end").collect()
last = RSE.sort("tick").group_by("match_key").agg(pl.col("t_score").last().alias("rs_t"),
                                                  pl.col("ct_score").last().alias("rs_ct"),
                                                  pl.col("round").max().alias("max_round"))
HH = H.join(last, on="match_key", how="inner").with_columns(
    ((pl.col("t_starters_score_final") == pl.col("rs_t")) & (pl.col("ct_starters_score_final") == pl.col("rs_ct"))).alias("same_order"),
    ((pl.col("t_starters_score_final") == pl.col("rs_ct")) & (pl.col("ct_starters_score_final") == pl.col("rs_t"))).alias("swap_order"),
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr_sum"),
)
print("matches with round_state round_end:", HH.height, "| builds:", HH.group_by("build_num").len().rows())
print(HH.group_by(["build_num", "same_order", "swap_order"]).len().sort(["build_num", "same_order", "swap_order"]))
print("hdr_sum vs max round_end count per match (build 9842), first 10:")
rc = RE.group_by("match_key").len().rename({"len": "n_round_end"})
print(HH.join(rc, on="match_key", how="left").select(
    ["match_key", "build_num", "t_starters_score_final", "ct_starters_score_final", "rs_t", "rs_ct",
     "max_round", "n_round_end"]).head(10))

print("\nDONE")
