"""verify D7-5 step 6: score the claimed fix against independent evidence.

Truth proxy (independent of round_end winner_team_code and of the score ledger):
  truth = bomb-derived side (exploded -> 2, defused -> 3) if a bomb outcome exists, else elimination side
  (side with all 5 distinct players dead inside the round window), else unknown.
Candidates for the round winner in build 9961 (5v5 rounds):
  winner   = round_end winner_team_code (as recorded)
  dp       = my free min-swap ledger (score deltas, D7-8 method)
  claim    = dp if not null, else bomb side if not null, else winner (the proposed fix; winner kept when neither exists)
Also reports bomb-vs-elim agreement as a validation of the truth proxy, and the rows changed by the claim fix.
Scratchpad R table only. Read-only.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_5")
R = pl.read_parquet(str(SCR / "R_all_rounds.parquet")).with_columns(
    pl.col("build_num").cast(pl.Utf8).alias("build"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round") == 25).then(pl.lit("r25"))
      .when(pl.col("round") == 27).then(pl.lit("r27_OT_last"))
      .when(pl.col("round") == 9).then(pl.lit("r09"))
      .when((pl.col("round") >= 28) & (((pl.col("round") - 25) % 3) == 0)).then(pl.lit("OT_fas_28plus"))
      .otherwise(pl.lit("other")).alias("cls"),
)
Q = R.filter(pl.col("player_count") == 20).filter(pl.col("winner_team_code").is_not_null())
S12 = Q.filter(pl.col("round") == 12).select(["match_key", "noswap_valid", "swap_valid"]).with_columns(
    pl.when(pl.col("noswap_valid") & pl.col("swap_valid")).then(pl.lit("both"))
      .when(pl.col("noswap_valid")).then(pl.lit("noswap_only"))
      .when(pl.col("swap_valid")).then(pl.lit("swap_only"))
      .otherwise(pl.lit("neither")).alias("pair12_type")).select(["match_key", "pair12_type"])
Q = Q.join(S12, on="match_key", how="left").with_columns(pl.col("pair12_type").fill_null("no_pair12"))
Q = Q.with_columns(
    pl.when(pl.col("bomb_side").is_not_null()).then(pl.col("bomb_side"))
      .when(pl.col("elim_side").is_not_null()).then(pl.col("elim_side"))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("truth"),
    pl.when(pl.col("bomb_side").is_not_null()).then(pl.lit("bomb"))
      .when(pl.col("elim_side").is_not_null()).then(pl.lit("elim"))
      .otherwise(pl.lit("none")).alias("truth_src"),
    pl.coalesce([pl.col("dp_side"), pl.col("bomb_side"), pl.col("winner_team_code")]).alias("claim"),
)
Q = Q.with_columns(pl.when(pl.col("pair12_type") == "noswap_only").then(pl.lit("degraded_noswap")).otherwise(pl.lit("normal")).alias("cluster"))

print("== validation: bomb-side vs elim-side agreement where both exist, by build and cluster ==")
BE = Q.filter(pl.col("bomb_side").is_not_null() & pl.col("elim_side").is_not_null())
print(BE.group_by(["build", "cluster"]).agg(pl.len().alias("n"),
                                            (pl.col("bomb_side") == pl.col("elim_side")).sum().alias("agree")).sort(["build", "cluster"]))

print("\n== truth coverage (9961 and 9842), by cluster ==")
print(Q.group_by(["build", "cluster", "truth_src"]).len().sort(["build", "cluster", "truth_src"]))

E = Q.filter(pl.col("truth").is_not_null())
print("\n== accuracy vs truth proxy, build 9961, normal cluster, by round class ==")
def acc(df):
    return df.group_by(["build", "cluster", "cls"]).agg(
        pl.len().alias("n_truth"),
        (pl.col("winner_team_code") == pl.col("truth")).sum().alias("acc_winner"),
        pl.col("dp_side").is_not_null().sum().alias("n_dp"),
        (pl.col("dp_side") == pl.col("truth")).sum().alias("acc_dp_where_known"),
        (pl.col("claim") == pl.col("truth")).sum().alias("acc_claim_fix"),
        (pl.col("claim") != pl.col("winner_team_code")).sum().alias("claim_changes_winner"),
    ).sort(["build", "cluster", "cls"])
print(acc(E.filter((pl.col("build") == "9961") & (pl.col("cluster") == "normal"))))

print("\n== accuracy vs truth proxy, build 9961, degraded noswap cluster ==")
print(acc(E.filter((pl.col("build") == "9961") & (pl.col("cluster") == "degraded_noswap"))))

print("\n== accuracy vs truth proxy, build 9842 (pre-change), by round class ==")
print(acc(E.filter(pl.col("build") == "9842")))

print("\n== totals on truth-covered rounds, by build x cluster ==")
print(E.group_by(["build", "cluster"]).agg(
    pl.len().alias("n_truth"),
    (pl.col("winner_team_code") == pl.col("truth")).sum().alias("acc_winner"),
    (pl.col("dp_side") == pl.col("truth")).sum().alias("acc_dp_known_rows"),
    pl.col("dp_side").is_not_null().sum().alias("n_dp_known"),
    (pl.col("claim") == pl.col("truth")).sum().alias("acc_claim_fix"),
    (pl.col("claim") != pl.col("winner_team_code")).sum().alias("claim_changes_winner"),
).sort(["build", "cluster"]))

print("\n== the claim fix applied to all 9961 5v5 rounds (no truth needed): rows whose label changes, by class ==")
C9961 = Q.filter(pl.col("build") == "9961")
print(C9961.filter(pl.col("claim") != pl.col("winner_team_code")).group_by(["cluster", "cls"]).len()
      .sort(["cluster", "cls"]).rows())
print("total 9961 5v5 rounds:", C9961.height, "| changed by claim fix:",
      C9961.filter(pl.col("claim") != pl.col("winner_team_code")).height,
      "| with no ledger and no bomb (kept as winner):", C9961.filter(pl.col("dp_side").is_null() & pl.col("bomb_side").is_null()).height)
