"""verify_D9_9 step 8: final clean numbers for the verdict (reads step-1 and step-3 scratchpad tables only).

Prints:
  - bomb-mismatch rows in build 9961 by round and format (should total 499 if the D9 side check is reproduced)
  - non-bomb build 9961 rounds at 13/25/28 (5v5) and 9 (2v2): claim rule vs score truth, precision/recall
  - non-bomb rounds where the score says recorded is correct but the death check passes only by 2v2 degeneracy
  - death-check pass rate for the true winner, 5v5 r13, 9842 vs 9961
  - the 2v2 composition of 9961 r13 non-bomb rounds where recorded agrees with score
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
R = pl.read_parquet(SCR / "verify_D9_9_round_oracle.parquet")
pat = (R.filter(pl.col("swap_sig")).sort(["match_key", "round"]).group_by("match_key")
       .agg(pl.col("round").cast(pl.Utf8).str.join(",").alias("swap_pat")))
fmt = (R.group_by("match_key").agg(pl.col("n_T").median().alias("nT_med"))
       .with_columns(pl.when(pl.col("nT_med") <= 2).then(pl.lit("2v2")).otherwise(pl.lit("5v5")).alias("fmt")))
R = R.join(pat, on="match_key", how="left").with_columns(pl.col("swap_pat").fill_null("none")).join(
    fmt.select(["match_key", "fmt"]), on="match_key", how="left")


def passes(w, df):
    return ((w == 2) & (df["CT_dead"] >= df["n_CT"])) | ((w == 3) & (df["T_dead"] >= df["n_T"]))


print("== F1. bomb-mismatch rows (bomb side != recorded winner) by build x format x round")
mm = R.filter(pl.col("has_outcome") & (pl.col("bomb_side") != pl.col("winner_team_code")))
print(mm.group_by(["build_num", "fmt", "round"]).len().sort(["build_num", "fmt", "round"]))
print("total bomb-mismatch rows, build 9961:", mm.filter(pl.col("build_num") == 9961).height,
      "| build 9842:", mm.filter(pl.col("build_num") == 9842).height,
      "| total all builds:", mm.height)
print("(D9 side check: 499 mismatches in 9961; claim population = pc20 subset; this is all bomb rounds)")

Q = R.filter(pl.col("build_num") == 9961).with_columns(
    passes(pl.col("winner_team_code"), R.filter(pl.col("build_num") == 9961)).alias("rec_pass") if False else pl.lit(None).alias("_x"))
Q = R.filter(pl.col("build_num") == 9961)
Q = Q.with_columns(
    (((pl.col("winner_team_code") == 2) & (pl.col("CT_dead") >= pl.col("n_CT"))) |
     ((pl.col("winner_team_code") == 3) & (pl.col("T_dead") >= pl.col("n_T")))).alias("rec_pass"),
    (((5 - pl.col("winner_team_code")) == 2) & (pl.col("CT_dead") >= pl.col("n_CT")) |
     ((5 - pl.col("winner_team_code")) == 3) & (pl.col("T_dead") >= pl.col("n_T"))).alias("flip_pass"),
)
NB = Q.filter(~pl.col("has_outcome") & pl.col("n_CT").is_not_null() & pl.col("w_score").is_not_null())
cr = NB.filter((pl.col("fmt") == "5v5") & pl.col("round").is_in([13, 25, 28]))
claim_flip = (~cr["rec_pass"]) & cr["flip_pass"]
inv = cr["w_score"] != cr["winner_team_code"]
print("\n== F2. claim rule (death-gated flip) vs score truth, build 9961, non-bomb 5v5 rounds 13/25/28")
print("   rounds:", cr.height, "| score says inverted:", int(inv.sum()))
print("   claim flips:", int(claim_flip.sum()), "| of those score-inverted (precision):",
      int((claim_flip & inv).sum()), "| score-inverted but not flipped (missed):", int((~claim_flip & inv).sum()))
print("   recall on score-inverted:", round(float((claim_flip & inv).sum()) / max(1, int(inv.sum())), 4))
print("   by round (inverted, flipped by claim, missed):")
for rr in [13, 25, 28]:
    m = cr["round"] == rr
    print("     r", rr, "inverted", int((inv & m).sum()), "claim-flipped", int((claim_flip & inv & m).sum()) if False else int(((claim_flip) & inv & m).sum()),
          "missed", int(((~claim_flip) & inv & m).sum()))

nine = NB.filter((pl.col("fmt") == "2v2") & (pl.col("round") == 9))
print("\n== F3. 2v2 round 9 (non-bomb, build 9961): not covered by the claim rule")
print("   rounds:", nine.height, "| score says inverted:", int((nine["w_score"] != nine["winner_team_code"]).sum()),
      "| death rec pass:", round(float(nine["rec_pass"].mean()), 4), "| death flip pass:", round(float(nine["flip_pass"].mean()), 4))

print("\n== F4. death-check pass rate for the TRUE winner (score truth), 5v5 round 13 non-bomb")
for b in [9842, 9961]:
    x = NB.filter((pl.col("build_num") == b) & (pl.col("fmt") == "5v5") & (pl.col("round") == 13)) if b == 9961 else \
        R.filter((pl.col("build_num") == b) & (~pl.col("has_outcome")) & pl.col("n_CT").is_not_null() & pl.col("w_score").is_not_null()
                 & (pl.col("fmt") == "5v5") & (pl.col("round") == 13))
    if b == 9961:
        truth_pass = x.filter(pl.col("w_score") != pl.col("winner_team_code"))["flip_pass"].mean()
        print("   9961 5v5 r13: score-inverted rounds", int((x["w_score"] != x["winner_team_code"]).sum()),
              "| true-winner (=flipped) death pass:", round(float(truth_pass), 4))
    else:
        xr = x.with_columns((((pl.col("winner_team_code") == 2) & (pl.col("CT_dead") >= pl.col("n_CT"))) |
                             ((pl.col("winner_team_code") == 3) & (pl.col("T_dead") >= pl.col("n_T")))).alias("rp"))
        print("   9842 5v5 r13: score-agrees rounds", xr.height, "| true-winner (=recorded) death pass:", round(float(xr["rp"].mean()), 4))

print("\n== F5. 9961 round 13 non-bomb: where the score says recorded is correct, by format")
ok13 = NB.filter((pl.col("round") == 13) & (pl.col("w_score") == pl.col("winner_team_code")))
print(ok13.group_by(["fmt", "swap_pat"]).len().sort(["fmt", "swap_pat"]))

print("\n== F6. round-27 oracle check (score-stream rule is unreliable at 27; bomb truth decides)")
r27 = R.filter((pl.col("round") == 27) & pl.col("has_outcome"))
print(r27.group_by("build_num").agg(pl.len().alias("n_bomb"),
                                    (pl.col("winner_team_code") == pl.col("bomb_side")).sum().alias("rec_eq_bomb")))
s27 = R.filter((pl.col("round") == 27) & pl.col("w_score").is_not_null())
print("   score-vs-recorded at 27 (9961, 9842):",
      s27.group_by("build_num").agg((pl.col("w_score") == pl.col("winner_team_code")).mean().round(3).alias("agree")).rows())
