"""verify_D9_9 step 7: compare fix rules against the two label-free truth sources.

Truth sources (independent of round_end winner):
  bomb truth  : bomb_exploded -> T (2), bomb_defused -> CT (3); bomb rounds only.
  score truth : in-game score stream winner (scripts 03/05 logic); non-bomb and bomb rounds with a score winner.
Format: '2v2' if the match's median n_T <= 2, else '5v5' (ASSUMPTION: roster from player_info).
Rules compared (build 9961 only; build 9842 controls shown as recorded):
  claim rule   : flip recorded at rounds {13,25,28} (5v5) only where recorded fails and flipped passes (death check).
  half rule    : flip recorded at first round of each half after the first: 5v5 {13,25,28}; 2v2 {9}.
  recorded     : no change.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(150)
pl.Config.set_tbl_cols(30)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
R = pl.read_parquet(SCR / "verify_D9_9_round_oracle.parquet")
pat = (R.filter(pl.col("swap_sig")).sort(["match_key", "round"]).group_by("match_key")
       .agg(pl.col("round").cast(pl.Utf8).str.join(",").alias("swap_pat")))
fmt = (R.group_by("match_key").agg(pl.col("n_T").median().alias("nT_med"))
       .with_columns(pl.when(pl.col("nT_med") <= 2).then(pl.lit("2v2")).otherwise(pl.lit("5v5")).alias("fmt")))
R = R.join(pat, on="match_key", how="left").with_columns(pl.col("swap_pat").fill_null("none")).join(
    fmt.select(["match_key", "fmt"]), on="match_key", how="left")

print("== (1) bomb truth by build x format x round (rounds 9, 12, 13, 14, 24, 25, 26, 27, 28, 29): n, recorded==bomb")
bo = R.filter(pl.col("has_outcome") & pl.col("round").is_in([9, 12, 13, 14, 24, 25, 26, 27, 28, 29]))
print(bo.group_by(["build_num", "fmt", "round"]).agg(
    pl.len().alias("n"),
    (pl.col("winner_team_code") == pl.col("bomb_side")).sum().alias("rec_eq_bomb"),
    (pl.col("bomb_side") == (5 - pl.col("winner_team_code"))).sum().alias("flip_eq_bomb"))
    .sort(["build_num", "fmt", "round"]))

print("\n== (2) score truth by build x format x round (non-bomb + bomb, score winner non-null): n, recorded==score")
sc = R.filter(pl.col("w_score").is_not_null() & pl.col("round").is_in([9, 12, 13, 14, 24, 25, 26, 27, 28, 29]))
print(sc.group_by(["build_num", "fmt", "round"]).agg(
    pl.len().alias("n"),
    (pl.col("winner_team_code") == pl.col("w_score")).mean().round(4).alias("rec_eq_score"),
    (pl.col("w_score") == (5 - pl.col("winner_team_code"))).mean().round(4).alias("flip_eq_score"))
    .sort(["build_num", "fmt", "round"]))

# rules on 9961 (apply to all rounds with a truth source)
def passes(w):
    return ((w == 2) & (pl.col("CT_dead") >= pl.col("n_CT"))) | ((w == 3) & (pl.col("T_dead") >= pl.col("n_T")))


Q = R.filter(pl.col("build_num") == 9961).with_columns(
    passes(pl.col("winner_team_code")).alias("rec_pass"),
    passes(5 - pl.col("winner_team_code")).alias("flip_pass"))
claim_flip = (pl.col("fmt") == "5v5") & pl.col("round").is_in([13, 25, 28]) & (~pl.col("rec_pass")) & pl.col("flip_pass") & (~pl.col("has_outcome"))
half_flip = ((pl.col("fmt") == "5v5") & pl.col("round").is_in([13, 25, 28])) | ((pl.col("fmt") == "2v2") & (pl.col("round") == 9))
Q = Q.with_columns(claim_flip.fill_null(False).alias("claim_flip"), half_flip.alias("half_flip"))
Q = Q.with_columns(
    pl.when(pl.col("half_flip")).then(5 - pl.col("winner_team_code")).otherwise(pl.col("winner_team_code")).alias("half_w"),
    pl.when(pl.col("claim_flip")).then(5 - pl.col("winner_team_code")).otherwise(pl.col("winner_team_code")).alias("claim_w"))

print("\n== (3) build 9961 rule accuracy vs BOMB truth (bomb rounds): rule winner == bomb side")
B = Q.filter(pl.col("has_outcome"))
print(B.group_by(["fmt", "round"]).agg(
    pl.len().alias("n"),
    (pl.col("winner_team_code") == pl.col("bomb_side")).sum().alias("recorded_ok"),
    (pl.col("claim_w") == pl.col("bomb_side")).sum().alias("claim_rule_ok"),
    (pl.col("half_w") == pl.col("bomb_side")).sum().alias("half_rule_ok"))
    .filter(pl.col("round").is_in([9, 13, 25, 28, 12, 14]) | (pl.col("fmt") == "2v2"))
    .sort(["fmt", "round"]))

print("\n== (4) build 9961 rule accuracy vs SCORE truth (rounds with score winner): rule winner == score winner")
S = Q.filter(pl.col("w_score").is_not_null())
print(S.group_by(["fmt", "round"]).agg(
    pl.len().alias("n"),
    (pl.col("winner_team_code") == pl.col("w_score")).sum().alias("recorded_ok"),
    (pl.col("claim_w") == pl.col("w_score")).sum().alias("claim_rule_ok"),
    (pl.col("half_w") == pl.col("w_score")).sum().alias("half_rule_ok"))
    .filter(pl.col("round").is_in([9, 13, 25, 28]))
    .sort(["fmt", "round"]))

print("\n== (5) claim rule: how many rounds it flips in 9961 (non-bomb), vs score truth")
cf = Q.filter(pl.col("claim_flip") & pl.col("w_score").is_not_null())
print("claim flips with score truth:", cf.height, "| of which score agrees with flip:",
      (cf["w_score"] == (5 - cf["winner_team_code"])).sum())
cf_all = Q.filter(pl.col("claim_flip"))
print("claim flips total (incl. no score):", cf_all.height)
print("claim rule: 5v5 non-bomb rounds at 13/25/28 still unflipped where score says inverted:")
un = Q.filter((pl.col("fmt") == "5v5") & pl.col("round").is_in([13, 25, 28]) & ~pl.col("has_outcome") & ~pl.col("claim_flip")
              & pl.col("w_score").is_not_null() & (pl.col("w_score") != pl.col("winner_team_code")))
print("   count:", un.height, "| by round:", un.group_by("round").len().sort("round").rows())

print("\n== (6) r9 in 9961 by format (non-bomb rounds): recorded pass rate and score agreement")
r9 = Q.filter((pl.col("round") == 9) & ~pl.col("has_outcome") & pl.col("n_CT").is_not_null())
print(r9.group_by(["fmt"]).agg(pl.len().alias("n"), pl.col("rec_pass").mean().round(4).alias("rec_pass"),
                              pl.col("flip_pass").mean().round(4).alias("flip_pass"),
                              (pl.col("w_score") == pl.col("winner_team_code")).mean().round(4).alias("score_eq_rec")))
print("\n== (7) round 27 check (last round before the OT switch): bomb truth and death check, by build")
r27 = R.filter((pl.col("round") == 27) & pl.col("has_outcome"))
print(r27.group_by("build_num").agg(pl.len().alias("n_bomb"), (pl.col("winner_team_code") == pl.col("bomb_side")).sum().alias("rec_eq_bomb")))
