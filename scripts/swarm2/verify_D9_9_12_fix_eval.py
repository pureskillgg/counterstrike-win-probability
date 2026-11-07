"""verify D9-9, step 12: evaluate the proposed death-based flip rule and the label-flip identity (read-only).

Uses the per-round table written by verify_D9_9_10 (three_source_round.parquet) and the boundary
status computed in verify_D9_9_11. Truth proxy for each round: BO (bomb outcome) where a bomb
outcome exists, otherwise SB (scoreboard winner). Only build 9961, non-bomb rounds 13, and 25/28
(claim scope), plus round 9 for the scope check.

Proposed rule (from the claim): for build-9961 non-bomb rounds 13/25/28, flip the recorded winner iff
the recorded side fails the death check and the flipped side passes; leave rounds where neither passes
as uncertain. Evaluated against the truth proxy.

Label-flip identity: pass(w) computed with all team labels (deaths and roster) swapped equals
pass(5 - w) computed with the original labels. Checked numerically on the same rounds.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9_9verify")
T = pl.read_parquet(SCR / "three_source_round.parquet").sort(["match_key", "round"])
T = T.with_columns(
    pl.col("t_post").shift(1).over("match_key").alias("pt"),
    pl.col("ct_post").shift(1).over("match_key").alias("pc"),
    pl.col("round").shift(1).over("match_key").alias("pr"),
).with_columns(
    ((pl.col("t_pre") == pl.col("pt")) & (pl.col("ct_pre") == pl.col("pc"))).alias("noswap_ok"),
    ((pl.col("t_pre") == pl.col("pc")) & (pl.col("ct_pre") == pl.col("pt"))).alias("swap_ok"),
).with_columns(
    pl.when(pl.col("pr") != pl.col("round") - 1).then(pl.lit("no_prev"))
      .when(pl.col("noswap_ok") & pl.col("swap_ok")).then(pl.lit("tie"))
      .when(pl.col("noswap_ok")).then(pl.lit("noswap"))
      .when(pl.col("swap_ok")).then(pl.lit("swap"))
      .otherwise(pl.lit("neither")).alias("bnd"))

# truth proxy
T = T.with_columns(
    pl.when(pl.col("has_bomb")).then(pl.col("BO")).otherwise(pl.col("SB")).alias("truth"))
B = T.filter((pl.col("build_num") == 9961) & pl.col("truth").is_not_null() & pl.col("n_CT").is_not_null()
             & (pl.col("n_CT") > 0) & (pl.col("n_T") > 0))

# label-flip identity: swapping all labels equals flipping the winner
Bf = B.with_columns(
    pl.col("CT_dead").alias("T_dead_f"), pl.col("T_dead").alias("CT_dead_f"),
    pl.col("n_CT").alias("n_T_f"), pl.col("n_T").alias("n_CT_f"))


def pass_with(df, w, td, ctd, nt, nct):
    return df.with_columns(
        (((pl.col(w) == 2) & (pl.col(ctd) >= pl.col(nct))) | ((pl.col(w) == 3) & (pl.col(td) >= pl.col(nt)))).alias("p"))


a = pass_with(Bf, "RE", "T_dead_f", "CT_dead_f", "n_T_f", "n_CT_f")["p"]
b = pass_with(B, "RE_flip", "T_dead", "CT_dead", "n_T", "n_CT")["p"]
NBr = B.filter(~pl.col("has_bomb") & (pl.col("build_num") == 9961) & pl.col("round").is_in([13, 25, 28]))
print("== label-flip identity on build 9961 round 13/25/28 non-bomb (n, recorded-with-flipped-labels, flipped-winner):")
ident = pass_with(Bf.filter(~pl.col("has_bomb") & pl.col("round").is_in([13, 25, 28])), "RE", "T_dead_f",
                  "CT_dead_f", "n_T_f", "n_CT_f").select(["p"]).rename({"p": "p_labels_flipped"})
ident2 = pass_with(B.filter(~pl.col("has_bomb") & pl.col("round").is_in([13, 25, 28])), "RE_flip", "T_dead",
                   "CT_dead", "n_T", "n_CT").select(["p"]).rename({"p": "p_winner_flipped"})
print("  rows:", ident.height, "| identical:", (ident["p_labels_flipped"] == ident2["p_winner_flipped"]).all())

# proposed rule on build 9961 non-bomb rounds 13/25/28
R = NBr.with_columns(
    pass_with(NBr, "RE", "T_dead", "CT_dead", "n_T", "n_CT")["p"].alias("ok_rec"),
    pass_with(NBr, "RE_flip", "T_dead", "CT_dead", "n_T", "n_CT")["p"].alias("ok_flp"),
).with_columns(
    pl.when(~pl.col("ok_rec") & pl.col("ok_flp")).then(pl.lit("flip"))
      .when(pl.col("ok_rec") & ~pl.col("ok_flp")).then(pl.lit("keep"))
      .when(pl.col("ok_rec") & pl.col("ok_flp")).then(pl.lit("both_pass"))
      .otherwise(pl.lit("uncertain")).alias("rule"),
    (pl.col("truth") != pl.col("RE")).alias("truth_flipped"),
)
print("\n== proposed death rule vs truth proxy (build 9961, non-bomb, rounds 13/25/28)")
print(R.group_by(["rule", "truth_flipped"]).agg(pl.len().alias("n")).sort(["rule", "truth_flipped"]))
summ = R.group_by("rule").agg(pl.len().alias("n"), pl.col("truth_flipped").mean().round(4).alias("share_truth_flipped"))
print(summ.sort("rule"))
tot = R.height
flip_ok = R.filter((pl.col("rule") == "flip") & pl.col("truth_flipped")).height
flip_n = R.filter(pl.col("rule") == "flip").height
print(f"\nflip rule precision (flipped rounds where truth is flipped): {flip_ok}/{flip_n}")
miss = R.filter((pl.col("rule") != "flip") & pl.col("truth_flipped")).height
print(f"truth-flipped rounds not flipped by the rule (missed): {miss} of {R.filter(pl.col('truth_flipped')).height}")
unc = R.filter(pl.col("rule") == "uncertain").height
print(f"uncertain (neither passes): {unc} of {tot} = {unc / tot:.4f}")

print("\n== round 9 scope check, build 9961 and 9842: boundary into 9 vs RE=SB and death pass (non-bomb)")
for bld in [9961, 9842]:
    X = T.filter((pl.col("build_num") == bld) & (pl.col("round") == 9) & ~pl.col("has_bomb") & pl.col("SB").is_not_null()
                 & pl.col("n_CT").is_not_null())
    X = X.with_columns((pl.col("RE") == pl.col("SB")).alias("RE_eq_SB"))
    print("build", bld)
    print(X.group_by(["bnd", "RE_eq_SB"]).agg(
        pl.len().alias("n"), pl.col("de_RE").mean().round(4).alias("pass_RE"),
        pl.col("de_SB").mean().round(4).alias("pass_SB")).sort(["bnd", "RE_eq_SB"]))

print("\n== bomb tick-window check, build 9961 round 13 and 25/28 (bomb outcome presence)")
print(T.filter((pl.col("build_num") == 9961) & pl.col("round").is_in([13, 25, 28]) & pl.col("has_bomb"))
      .group_by(["round", "BO", "SB", "RE"]).len().sort(["round", "BO", "SB", "RE"]))
