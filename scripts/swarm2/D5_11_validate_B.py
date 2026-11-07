"""D5 step 11: explain the regime-B residual in the score-lead reconstruction (D5_10 part 3).

For regime-B matches with a nonzero team-A difference, check:
  (a) whether freeze-end round 2 is missing (so the round-1 inference is null),
  (b) whether any round label r in 1..12 has a missing freeze-end (fe) row or a missing round_end row,
  (c) whether the valid freeze-end step (r -> r+1, no side swap, labels 2..11) disagrees with the
      round_end winner in those matches.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_11_validate_B.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
mk = R.group_by("match_key").agg(
    pl.col("n_round_announce_match_start").fill_null(0).sum().alias("mstart"))
B = (M.select(["match_key", "day"]).join(mk, on="match_key", how="left")
     .filter(pl.col("mstart") == 0).select("match_key"))
RB = R.join(B, on="match_key", how="inner").sort(["match_key", "round"])
print("regime B matches:", B.height)

fe2 = RB.filter(pl.col("round") == 2).select(["match_key", pl.col("fe_t_score").alias("t2"), pl.col("fe_ct_score").alias("ct2")])
print("regime B matches with no freeze-end at round 2:", B.height - fe2.filter(pl.col("t2").is_not_null()).height)

# missing rows by label 1..12 in regime B
miss = RB.filter(pl.col("round").is_between(1, 12)).group_by("round").agg(
    pl.len().alias("rows"),
    (~pl.col("in_end")).sum().alias("no_round_end"),
    (~pl.col("in_fe")).sum().alias("no_freeze_end")).sort("round")
print("\nregime B: missing rows by label 1..12 (out of matches):")
print(miss)

# valid freeze-end step vs round_end winner, labels 2..11 (no side swap in between)
fe = RB.select(["match_key", "round", "fe_t_score", "fe_ct_score", "winner"])
step = (fe.with_columns(nt=pl.col("fe_t_score").shift(-1).over("match_key"),
                        nct=pl.col("fe_ct_score").shift(-1).over("match_key"),
                        nr=pl.col("round").shift(-1).over("match_key"))
        .filter((pl.col("nr") == pl.col("round") + 1) & pl.col("round").is_between(2, 11)
                & pl.col("fe_t_score").is_not_null() & pl.col("nt").is_not_null() & pl.col("winner").is_not_null())
        .with_columns(dt=pl.col("nt") - pl.col("fe_t_score"), dct=pl.col("nct") - pl.col("fe_ct_score"))
        .with_columns(ch=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
                      .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0))))
print("\nregime B valid step (labels 2..11) vs round_end winner:")
print(step.group_by("round").agg(pl.len().alias("steps"), (pl.col("ch") == pl.col("winner")).sum().alias("agree"),
                                 (pl.col("ch") == 0).sum().alias("chain_zero")).sort("round"))

# which regime-B matches have nonzero diff (use D5_10 logic) and do they have missing rows?
def team_a_is_t(col):
    return (pl.col(col) <= 12) | pl.col(col).is_between(25, 27)


RX = RB.join(fe2, on="match_key", how="left")
RX = RX.with_columns(
    a_fe=pl.when(team_a_is_t("round")).then(pl.col("fe_t_score")).otherwise(pl.col("fe_ct_score")),
    a_won=pl.when((pl.col("round") == 1)).then(pl.when(team_a_is_t("round")).then(pl.col("t2")).otherwise(pl.col("ct2")))
    .otherwise(pl.when(team_a_is_t("round") & (pl.col("winner") == 2)).then(1)
               .when((~team_a_is_t("round")) & (pl.col("winner") == 3)).then(1).otherwise(0)))
RX = RX.with_columns(cum_before=pl.col("a_won").fill_null(0).cum_sum().over("match_key") - pl.col("a_won").fill_null(0))
RX = RX.with_columns(diff=pl.col("a_fe") - pl.col("cum_before"))
bad = RX.filter((pl.col("diff") != 0) & pl.col("a_fe").is_not_null()).select("match_key").unique()
print("\nregime B matches with nonzero diff:", bad.height)
badm = RB.join(bad, on="match_key", how="inner")
print("  of these, with round-2 freeze-end missing:", bad.join(fe2.filter(pl.col("t2").is_null()), on="match_key", how="inner").height)
print("  of these, with any label 1..12 missing round_end or freeze_end:",
      badm.filter(pl.col("round").is_between(1, 12) & (~pl.col("in_end") | ~pl.col("in_fe")))["match_key"].n_unique())
print("  of these, with any label 2..11 chain disagreement:",
      step.join(bad, on="match_key", how="inner").filter(pl.col("ch") != pl.col("winner"))["match_key"].n_unique())
print("  of the rest (no missing rows, no chain disagreement):",
      bad.height
      - bad.join(fe2.filter(pl.col("t2").is_null()), on="match_key", how="inner").height)
