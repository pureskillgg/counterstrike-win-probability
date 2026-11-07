"""verify D7-5 step 8: is a fixed structural schedule a better ledger than the min-swap DP?

Schedule orientation (ASSUMPTION, standard CS rules): sides swap before round 13 (halftime), before round 25 (OT start),
and before rounds 28, 31, 34, ... (OT halves of 3 rounds). Round side comes from the freeze_end score deltas under that
schedule (sch_side), with no minimum-swap search.
Truth proxy: bomb-derived side if present, else elimination side (as in step 6). Scratchpad R table only.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_5")
R = pl.read_parquet(str(SCR / "R_all_rounds.parquet")).with_columns(
    pl.col("build_num").cast(pl.Utf8).alias("build"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round") == 25).then(pl.lit("r25"))
      .when(pl.col("round") == 27).then(pl.lit("r27_OT_last"))
      .when(pl.col("round") == 9).then(pl.lit("r09"))
      .when((pl.col("round") >= 28) & (((pl.col("round") - 25) % 3) == 0)).then(pl.lit("OT_fas_28plus"))
      .otherwise(pl.lit("other")).alias("cls"))
Q = R.filter((pl.col("player_count") == 20) & pl.col("winner_team_code").is_not_null())
S12 = Q.filter(pl.col("round") == 12).select(["match_key", "noswap_valid", "swap_valid"]).with_columns(
    pl.when(pl.col("noswap_valid") & pl.col("swap_valid")).then(pl.lit("both"))
      .when(pl.col("noswap_valid")).then(pl.lit("noswap_only"))
      .when(pl.col("swap_valid")).then(pl.lit("swap_only"))
      .otherwise(pl.lit("neither")).alias("pair12_type")).select(["match_key", "pair12_type"])
Q = Q.join(S12, on="match_key", how="left").with_columns(pl.col("pair12_type").fill_null("no_pair12"))
Q = Q.with_columns(
    pl.when(pl.col("pair12_type") == "noswap_only").then(pl.lit("degraded_noswap")).otherwise(pl.lit("normal")).alias("cluster"),
    pl.when(pl.col("bomb_side").is_not_null()).then(pl.col("bomb_side"))
      .when(pl.col("elim_side").is_not_null()).then(pl.col("elim_side"))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("truth"))
E = Q.filter(pl.col("truth").is_not_null() & (pl.col("cluster") == "normal"))
print("== normal cluster, truth-covered 5v5 rounds: accuracy by build x class (winner vs DP vs schedule) ==")
print(E.group_by(["build", "cls"]).agg(
    pl.len().alias("n_truth"),
    (pl.col("winner_team_code") == pl.col("truth")).sum().alias("acc_winner"),
    pl.col("dp_side").is_not_null().sum().alias("n_dp"),
    (pl.col("dp_side") == pl.col("truth")).sum().alias("acc_dp_known"),
    pl.col("sch_side").is_not_null().sum().alias("n_sch"),
    (pl.col("sch_side") == pl.col("truth")).sum().alias("acc_sch_known"),
).sort(["build", "cls"]))
T = E.group_by("build").agg(
    pl.len().alias("n_truth"),
    (pl.col("winner_team_code") == pl.col("truth")).sum().alias("acc_winner"),
    (pl.col("sch_side") == pl.col("truth")).sum().alias("acc_sch_known"),
    pl.col("sch_side").is_not_null().sum().alias("n_sch"),
    (pl.col("dp_side") == pl.col("truth")).sum().alias("acc_dp_known"),
    pl.col("dp_side").is_not_null().sum().alias("n_dp"),
).sort("build")
print("\ntotals (normal cluster, truth-covered):")
print(T)
print("\nschedule-invalid rounds (sch_valid false) by build and round class:")
print(Q.filter(pl.col("sch_valid") == False).group_by(["build", "pair12_type", "cls"]).len().sort(["build", "cls"]).rows())
print("\nrounds where schedule side != DP side, by build x class (normal cluster):")
print(Q.filter((pl.col("cluster") == "normal") & pl.col("sch_side").is_not_null() & pl.col("dp_side").is_not_null())
      .group_by(["build", "cls"]).agg((pl.col("sch_side") != pl.col("dp_side")).sum().alias("sch_ne_dp"), pl.len().alias("n"))
      .sort(["build", "cls"]))
