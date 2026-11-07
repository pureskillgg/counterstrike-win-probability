"""verify_D1_7 diagnostics (swarm2 verifier).

Reuses the table builder from verify_D1_7_recompute.py (same file directory) to answer:
  Q1 Is the claimed disagreement exactly 'swap inside the round_state window AND pre-round tie'?
     (crosstab over claimed-rule non-null rows, and over all rows, for rounds 27/33/39/45)
  Q2 Which rounds carry a mid-window swap signature at all (rounds 1..50), both tomes?
  Q3 Are the claimant's extra 2024 disagreements (rounds 9, 13, 25, 28) explained by the same swap window?
  Q4 For 2024 round 28 (first round after the OT swap): how do bomb, game win message, winner and the
     pre-swap scoreboard line up?
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2")))
import verify_D1_7_recompute as R  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)

hdr = R.hdr


def q1(A, label):
    hdr(f"Q1 {label}: mechanism crosstab for claimed rounds")
    for r in [27, 33, 39, 45]:
        x = A.filter(pl.col("round") == r)
        if x.height == 0:
            print(f"round {r}: no rows")
            continue
        print(f"\nround {r}: rows={x.height}")
        print("  claim non-null rows by (swap_sig, pre_tie, claim_dis):")
        cn = x.filter(pl.col("claim_side").is_not_null())
        print("   ", cn.group_by(["swap_sig", "pre_tie", "claim_dis"]).len().sort(["swap_sig", "pre_tie", "claim_dis"]).to_dicts())
        print("  claim null rows by (swap_sig, pre_tie):")
        cnull = x.filter(pl.col("claim_side").is_null())
        print("   ", cnull.group_by(["swap_sig", "pre_tie"]).len().sort(["swap_sig", "pre_tie"]).to_dicts())
        print("  corrected rule: non-null", x.filter(pl.col("corr_side").is_not_null()).height,
              " disagreements", x.filter(pl.col("corr_dis")).height)
        print("  bomb-verified:", x.filter(pl.col("bomb_side").is_not_null()).height,
              " winner-bomb disagreements:", x.filter(pl.col("bomb_dis")).height,
              " corrected-score vs bomb disagreements:",
              x.filter(pl.col("bomb_side").is_not_null() & pl.col("corr_side").is_not_null() & (pl.col("corr_side") != pl.col("bomb_side"))).height)
        print("  win-message rows:", x.filter(pl.col("msg_side").is_not_null()).height,
              " winner-message disagreements:", x.filter(pl.col("msg_dis")).height)


def q2(A, label):
    hdr(f"Q2 {label}: rounds with mid-window swap signature (all rounds 1..50)")
    t = (A.group_by("round").agg(pl.len().alias("rows"), pl.col("swap_sig").sum().alias("swap_sig"),
                                 pl.col("swap_tie").sum().alias("swap_tie"),
                                 pl.col("claim_dis").sum().alias("claim_dis"),
                                 pl.col("corr_dis").sum().alias("corr_dis"),
                                 pl.col("claim_side").is_not_null().sum().alias("claim_n"))
          .filter(pl.col("round") <= 50).sort("round"))
    print(t.filter((pl.col("swap_sig") > 0) | (pl.col("swap_tie") > 0)))


def q3(A, label):
    hdr(f"Q3 {label}: where do the non-swap disagreements sit (rounds 9, 13, 25, 28), claim vs corrected vs bomb")
    for r in [9, 13, 25, 28]:
        x = A.filter(pl.col("round") == r).with_columns(
            pl.when(pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)])).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
        print(f"\nround {r}:")
        print(x.group_by(["dg", "swap_sig"]).agg(
            pl.len().alias("rows"),
            pl.col("corr_side").is_not_null().sum().alias("corr_n"),
            pl.col("corr_dis").sum().alias("corr_dis"),
            pl.col("claim_side").is_not_null().sum().alias("claim_n"),
            pl.col("claim_dis").sum().alias("claim_dis"),
            pl.col("bomb_side").is_not_null().sum().alias("bomb_n"),
            pl.col("bomb_dis").sum().alias("bomb_dis"),
            pl.col("msg_dis").sum().alias("msg_dis")).sort(["dg", "swap_sig"]))


def q4(A, label):
    hdr(f"Q4 {label}: round 28 cross-source table (bomb side, message side, winner, pre-swap score side)")
    x = A.filter(pl.col("round") == 28).with_columns(
        pl.when(pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)])).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
    xb = x.filter(pl.col("bomb_side").is_not_null())
    print("bomb rows:", xb.height)
    print(xb.group_by(["dg", "bomb_side", "msg_side", "winner_team_code", "corr_side"]).len()
          .sort(["dg", "bomb_side", "msg_side", "winner_team_code", "corr_side"]))
    print("\nall rows (no bomb needed): winner vs message vs corrected score")
    print(x.group_by(["dg", "winner_team_code", "msg_side", "corr_side"]).len().sort(["dg", "winner_team_code", "msg_side", "corr_side"]))


def run():
    A24 = R.build(R.T24, "2024 tome", 20)
    A26 = R.build(R.T26, "2026 tome", 10)
    q1(A24, "2024 pc20")
    q1(A26, "2026 pc10")
    q2(R.build(R.T24, "2024 tome, all pc", None), "2024 all pc")
    q2(R.build(R.T26, "2026 tome, all pc", None), "2026 all pc")
    q3(A24, "2024 pc20")
    q4(A24, "2024 pc20")


if __name__ == "__main__":
    run()
