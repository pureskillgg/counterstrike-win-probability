"""D5 step 10: two recording regimes in the 2024 tome, the per-match marker, and score-lead impact.

Regime A  round_announce_match_start present (count >= 1), cs_win_panel_round absent,
          round_end label 1 present, round_state round_start/round_end absent.
Regime B  cs_win_panel_round present, round_announce_match_start absent, round_end label 1 absent.
Checks:
  1. Per-match crosstab of the markers (mstart, cs_win_panel_round, r1_end, grp).
  2. Duplicate round_end match (2f5bcb89...): regime, labels, freeze-end scores.
  3. Score-lead reconstruction in regime A: cumulative team-A score from round_end winners
     (team A = the team that starts on T; team A is T in halves H1 and OT1, CT in H2 and OT2)
     vs the freeze-end score of team A (t_score in H1/OT1, ct_score in H2/OT2). diff = fe - cum.
     ASSUMPTION: field swaps at 12->13, 24->25 and 27->28 (observed in D5_05), score fields side-based.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_10_regime.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
M = M.with_columns(grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))

mk = (R.group_by("match_key").agg(
    pl.col("n_round_announce_match_start").fill_null(0).sum().alias("mstart"),
    pl.col("n_cs_win_panel_round").fill_null(0).sum().alias("cs_win_panel_round"),
    pl.col("n_round_start").fill_null(0).sum().alias("rs_state_round_start"),
    pl.col("n_round_end").fill_null(0).sum().alias("rs_state_round_end"),
    pl.col("in_end").filter(pl.col("round") == 1).any().alias("r1_end")))
X = M.select(["match_key", "grp", "day", "aff_r13"]).join(mk, on="match_key", how="left")
X = X.with_columns(pl.col("r1_end").fill_null(False))

print("== 1. per-match marker crosstab ==")
X = X.with_columns(mstart_pos=(pl.col("mstart") > 0), cwp_pos=(pl.col("cs_win_panel_round") > 0),
                   rs_pos=(pl.col("rs_state_round_start") > 0))
print(X.group_by(["grp", "mstart_pos", "cwp_pos", "rs_pos", "r1_end"]).agg(pl.len().alias("matches"))
      .sort(["grp", "mstart_pos", "cwp_pos", "rs_pos", "r1_end"]))
print("\nregime by marker (regime A <=> round_announce_match_start present):")
Y = X.with_columns(regime=pl.when(pl.col("mstart") > 0).then(pl.lit("A")).otherwise(pl.lit("B")))
print(Y.group_by(["regime", "r1_end", "grp"]).agg(pl.len().alias("matches")).sort(["regime", "grp"]))
print("mismatch of marker vs r1_end:", Y.filter((pl.col("regime") == "A") != pl.col("r1_end")).height)
print("mismatch of cs_win_panel_round>0 vs regime B:",
      Y.filter((pl.col("cs_win_panel_round") > 0) != (pl.col("regime") == "B")).height)

print("\n== 2. duplicate round_end match ==")
dm = R.filter(pl.col("n_end_rows") > 1)["match_key"].unique().to_list()
print("match:", dm)
print(Y.filter(pl.col("match_key").is_in(dm)).select(["match_key", "grp", "regime", "mstart", "cs_win_panel_round"]))
print(R.filter(pl.col("match_key").is_in(dm)).sort("round").filter(pl.col("round") <= 3).select(
    ["round", "fe_tick", "fe_t_score", "fe_ct_score", "end_tick", "winner", "reason_code", "n_end_rows", "n_round_announce_match_start"]))

print("\n== 3. score-lead reconstruction: cumulative team-A score from round_end winners vs freeze-end team-A score ==")
print("regime A uses round_end label 1; regime B has no round_end label 1, so its round-1 result is taken from fe(2) (score before round 2, which starts 0-0).")


def team_a_is_t(col):
    # H1 (1..12) and OT1 (25..27): team A on T; H2 (13..24) and OT2 (28..30): team A on CT
    return (pl.col(col) <= 12) | pl.col(col).is_between(25, 27)


fe2 = R.filter(pl.col("round") == 2).select(["match_key", pl.col("fe_t_score").alias("t2"), pl.col("fe_ct_score").alias("ct2")])
RX = (R.join(Y.select(["match_key", "grp", "regime"]), on="match_key", how="inner")
      .join(fe2, on="match_key", how="left").sort(["match_key", "round"]))
RX = RX.with_columns(
    a_is_t=team_a_is_t("round"),
    a_fe=pl.when(team_a_is_t("round")).then(pl.col("fe_t_score")).otherwise(pl.col("fe_ct_score")))
# team-A win indicator per round from round_end winner; for regime B round 1 use fe(2) instead
RX = RX.with_columns(
    a_won_rend=pl.when(team_a_is_t("round") & (pl.col("winner") == 2)).then(1)
    .when((~team_a_is_t("round")) & (pl.col("winner") == 3)).then(1).otherwise(0),
    a_fe2_round1=pl.when(team_a_is_t("round")).then(pl.col("t2")).otherwise(pl.col("ct2")))
RX = RX.with_columns(
    a_won=pl.when((pl.col("round") == 1) & (pl.col("regime") == "B")).then(pl.col("a_fe2_round1"))
    .otherwise(pl.col("a_won_rend")))
RX = RX.with_columns(cum_before=pl.col("a_won").cum_sum().over("match_key") - pl.col("a_won"))
RX = RX.with_columns(diff=pl.col("a_fe") - pl.col("cum_before"))
ck = RX.filter(pl.col("a_fe").is_not_null() & (pl.col("round") >= 2))
ck = ck.filter(pl.col("winner").is_not_null() | (pl.col("regime") == "B"))
for reg in ["A", "B"]:
    c = ck.filter(pl.col("regime") == reg)
    print(f"\n-- regime {reg}: states (round >= 2, freeze-end present) = {c.height}; diff == 0: "
          f"{c.filter(pl.col('diff') == 0).height}; diff != 0: {c.filter(pl.col('diff') != 0).height}")
    print("   matches with any diff != 0:", c.filter(pl.col("diff") != 0)["match_key"].n_unique(),
          "of", c["match_key"].n_unique())
    print("   diff value counts:")
    print(c.group_by("diff").agg(pl.len().alias("states")).sort("diff"))
    print("   diff != 0 by round label:")
    print(c.filter(pl.col("diff") != 0).group_by("round").agg(pl.len().alias("states")).sort("round").head(40))
    print("   diff != 0 in H1 (1..12), H2 (13..24), OT (25..30):")
    print(c.with_columns(half=pl.when(pl.col("round") <= 12).then(pl.lit("H1"))
                         .when(pl.col("round") <= 24).then(pl.lit("H2")).otherwise(pl.lit("OT")))
          .group_by("half").agg(pl.len().alias("states"), (pl.col("diff") != 0).sum().alias("diff_nonzero")).sort("half"))
