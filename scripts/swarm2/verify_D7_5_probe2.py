"""verify D7-5 step 3: follow-up probes for the alternatives found in step 2.

(a) storage path day (csds/2024/02/DD) vs header match_date and build: is the day-by-day pattern a mixing artifact?
(b) round 9 and round 27 (non-structural rounds where winner_code and DP ledger disagree): which side does
    elimination / bomb evidence support, by build?
(c) the 9961 halftime pairs with no swap signature in the raw score fields (noswap_only at pair 12->13):
    what does elimination / bomb evidence say about round 13 for those matches?
(d) round-13 winner vs elimination evidence, split by halftime signature, for build 9961.
Reads the scratchpad R table only (raw-derived). Read-only with respect to the tome.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(220)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_5")
R0 = pl.read_parquet(str(SCR / "R_all_rounds.parquet"))
R = R0.with_columns(
    pl.col("build_num").cast(pl.Utf8).alias("build"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("path_day"),
    (pl.col("mdt") >= pl.datetime(2024, 2, 7, 1, 1, 18)).alias("mdt_after_onset"),
)
Q = R.filter(pl.col("player_count") == 20)

print("== (a) path day x build, 5v5 matches (header mdt vs path day) ==")
M = Q.group_by("match_key").agg(pl.col("build").first(), pl.col("path_day").first(),
                                pl.col("mdt").first(), pl.col("mdt_after_onset").first())
print(M.group_by(["path_day", "build"]).len().sort(["path_day", "build"]).rows())
print("matches whose header mdt is before onset but path day >= 12:",
      M.filter((pl.col("path_day") >= "12") & (~pl.col("mdt_after_onset"))).height)
print("build 9842 matches with path day >= 12:", M.filter((pl.col("build") == "9842") & (pl.col("path_day") >= "12")).height)
print("build 9961 matches with path day <= 07:", M.filter((pl.col("build") == "9961") & (pl.col("path_day") <= "07")).height)

print("\n== (a2) round-13 bomb rows by path day and build (winner vs bomb) ==")
B13 = Q.filter((pl.col("round") == 13) & pl.col("bomb_side").is_not_null())
print(B13.group_by(["path_day", "build"]).agg(
    pl.len().alias("n"),
    (pl.col("winner_team_code") == pl.col("bomb_side")).sum().alias("win_eq_bomb"),
    (pl.col("dp_side") == pl.col("bomb_side")).sum().alias("dp_eq_bomb"),
).sort(["path_day", "build"]))

print("\n== (b) rounds 8, 9, 13, 27 (5v5): winner vs elim, dp vs elim, winner vs bomb, dp vs bomb, by build ==")
for rd in [8, 9, 12, 13, 14, 26, 27, 28]:
    S = Q.filter(pl.col("round") == rd)
    out = S.group_by("build").agg(
        pl.len().alias("n"),
        pl.col("dp_side").is_not_null().sum().alias("n_dp"),
        (pl.col("winner_team_code") != pl.col("dp_side")).sum().alias("win_ne_dp"),
        pl.col("elim_side").is_not_null().sum().alias("n_elim"),
        (pl.col("winner_team_code") == pl.col("elim_side")).sum().alias("win_eq_elim"),
        (pl.col("dp_side") == pl.col("elim_side")).sum().alias("dp_eq_elim"),
        pl.col("bomb_side").is_not_null().sum().alias("n_bomb"),
        (pl.col("winner_team_code") == pl.col("bomb_side")).sum().alias("win_eq_bomb"),
        (pl.col("dp_side") == pl.col("bomb_side")).sum().alias("dp_eq_bomb"),
    ).sort("build")
    print(f"round {rd}:")
    print(out)

print("\n== (c) halftime signature at pair 12->13 (raw freeze_end scores), 9961 matches: round-13 evidence ==")
H12 = Q.filter(pl.col("round") == 12).select(["match_key", "build", "noswap_valid", "swap_valid"]).rename(
    {"noswap_valid": "ns12", "swap_valid": "sw12"})
R13 = Q.filter(pl.col("round") == 13)
J = R13.join(H12, on=["match_key", "build"], how="left").with_columns(
    pl.when(pl.col("ns12") & pl.col("sw12")).then(pl.lit("both"))
      .when(pl.col("ns12")).then(pl.lit("noswap_only"))
      .when(pl.col("sw12")).then(pl.lit("swap_only"))
      .otherwise(pl.lit("neither")).alias("pair12_type"))
print("round-13 rows by build x pair12 type x (winner==dp):")
print(J.group_by(["build", "pair12_type"]).agg(
    pl.len().alias("n"),
    pl.col("dp_side").is_not_null().sum().alias("n_dp"),
    (pl.col("winner_team_code") == pl.col("dp_side")).sum().alias("win_eq_dp"),
    pl.col("elim_side").is_not_null().sum().alias("n_elim"),
    (pl.col("winner_team_code") == pl.col("elim_side")).sum().alias("win_eq_elim"),
    (pl.col("dp_side") == pl.col("elim_side")).sum().alias("dp_eq_elim"),
    pl.col("bomb_side").is_not_null().sum().alias("n_bomb"),
    (pl.col("winner_team_code") == pl.col("bomb_side")).sum().alias("win_eq_bomb"),
    (pl.col("dp_side") == pl.col("bomb_side")).sum().alias("dp_eq_bomb"),
).sort(["build", "pair12_type"]))

print("\n== (d) 9961 round-13 rows with elimination evidence: winner vs elim, split by dp vs winner ==")
D = J.filter((pl.col("build") == "9961") & pl.col("elim_side").is_not_null() & pl.col("dp_side").is_not_null())
print(D.group_by(["pair12_type", "w_eq_dp", "w_eq_elim", "dp_eq_elim"]).len().sort(["pair12_type", "w_eq_dp"]).rows()
      if "w_eq_dp" in D.columns else
      D.with_columns((pl.col("winner_team_code") == pl.col("dp_side")).alias("w_eq_dp"),
                     (pl.col("winner_team_code") == pl.col("elim_side")).alias("w_eq_elim"),
                     (pl.col("dp_side") == pl.col("elim_side")).alias("dp_eq_elim"))
      .group_by(["pair12_type", "w_eq_dp", "w_eq_elim", "dp_eq_elim"]).len()
      .sort(["pair12_type", "w_eq_dp", "w_eq_elim", "dp_eq_elim"]).rows())

print("\n== (e) 9961 halftime pairs with no swap signature: sample raw scores around round 11-14 ==")
ns_matches = J.filter((pl.col("build") == "9961") & (pl.col("pair12_type") == "noswap_only"))["match_key"].head(3).to_list()
S = R0.filter(pl.col("match_key").is_in(ns_matches) & pl.col("round").is_between(10, 15)).select(
    ["match_key", "round", "fe_tick", "t_score", "ct_score", "winner_team_code", "elim_side", "dp_side", "bomb_side"])
print(S.with_columns(pl.col("match_key").str.slice(0, 22)).sort(["match_key", "round"]))
