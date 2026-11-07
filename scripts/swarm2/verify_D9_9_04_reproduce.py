"""verify_D9_9 step 4: (A) reproduce the claim's death-check numbers with the claim's definitions,
(B) cross the death check with the score-stream oracle, by build x round class x match swap pattern.

Claim definitions (D9_tests4/5/8):
  population: no bomb outcome row in the round (bomb_defused or bomb_exploded), n_CT not null (player_info roster).
  recorded passes: winner 2 and CT_dead >= n_CT, or winner 3 and T_dead >= n_T.
  flipped passes: same rule with winner = 5 - recorded.
  victim side = player_death.player_team_code (null victim rows are not counted), roster = player_info team_code,
  deduplicated per (match, round, player) with keep = last.
"""
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
R = pl.read_parquet(SCR / "verify_D9_9_round_oracle.parquet")
M = pl.read_parquet(SCR / "verify_D9_9_match.parquet")

# per-match swap pattern (score-stream), attached to every round of the match
R = R.with_columns(
    pl.col("swap_sig").cast(pl.Int64).alias("swap_sig_i"))
pat = (R.filter(pl.col("swap_sig")).sort(["match_key", "round"])
       .group_by("match_key").agg(pl.col("round").cast(pl.Utf8).str.join(",").alias("swap_pat")))
R = R.join(pat, on="match_key", how="left").with_columns(pl.col("swap_pat").fill_null("none"))


def passes(w, ct_dead, t_dead, n_ct, n_t):
    return ((w == 2) & (ct_dead >= n_ct)) | ((w == 3) & (t_dead >= n_t))


R = R.with_columns(
    passes(pl.col("winner_team_code"), pl.col("CT_dead"), pl.col("T_dead"), pl.col("n_CT"), pl.col("n_T")).alias("rec_pass"),
    passes(5 - pl.col("winner_team_code"), pl.col("CT_dead"), pl.col("T_dead"), pl.col("n_CT"), pl.col("n_T")).alias("flip_pass"),
)
NB = R.filter(~pl.col("has_outcome") & pl.col("n_CT").is_not_null())

print("== (A) reproduce claim: non-bomb rounds with roster, recorded vs flipped death-check pass rate")
print(NB.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n"),
    pl.col("rec_pass").mean().round(4).alias("rate_recorded"),
    pl.col("flip_pass").mean().round(4).alias("rate_flipped")).sort(["build_num", "rclass"]))
print("   claim: 9961 r13 n=1838 rec 0.1077 flip 0.8689 | 9961 r25_28 n=192 rec 0.0 flip 0.8958 | "
      "9961 other n=36324 rec 0.9412 | 9842 r13 n=1473 rec 0.9518 | 9842 r25_28 n=183 rec 0.9617")
print("\n   per round index (9961, recorded pass rate), rounds 1-30:")
g = NB.filter(pl.col("build_num") == 9961).group_by("round").agg(pl.len().alias("n"), pl.col("rec_pass").mean().round(2).alias("rec")).sort("round")
print("   " + "  ".join(f"r{r}:({n},{v})" for r, n, v in g.select(["round", "n", "rec"]).rows()))

print("\n== (B) cross: death check vs score oracle, non-bomb rounds, by build x round class")
NBs = NB.filter(pl.col("w_score").is_not_null())
print(NBs.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n"),
    pl.col("score_eq_recorded").mean().round(4).alias("score_eq_recorded"),
    (pl.col("w_score") == (5 - pl.col("winner_team_code"))).mean().round(4).alias("score_eq_flipped"),
    pl.col("rec_pass").mean().round(4).alias("death_rec_pass"),
    pl.col("flip_pass").mean().round(4).alias("death_flip_pass")).sort(["build_num", "rclass"]))

print("\n== (B2) 9961 round 13, non-bomb: score oracle vs recorded, by match swap pattern")
r13 = NBs.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13))
print(r13.group_by("swap_pat").agg(pl.len().alias("n"), pl.col("score_eq_recorded").mean().round(3).alias("score_eq_rec"),
                                   pl.col("rec_pass").mean().round(3).alias("death_rec_pass")).sort("n", descending=True))
print("\n== (B3) 9842 round 13, non-bomb: score oracle vs recorded, by swap pattern")
r13b = NBs.filter((pl.col("build_num") == 9842) & (pl.col("round") == 13))
print(r13b.group_by("swap_pat").agg(pl.len().alias("n"), pl.col("score_eq_recorded").mean().round(3).alias("score_eq_rec")).sort("n", descending=True))
