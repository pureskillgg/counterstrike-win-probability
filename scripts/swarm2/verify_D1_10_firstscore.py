"""verify_D1_10_firstscore.py -- fourth check for claim D1-10 (2026 tome), read-only, stdout only.

Rule under test (ASSUMPTION A6): the round's scoring event is the EARLIEST score_update row (by tick) in the
round whose new_score - old_score == +1. Later same-round reassignment rows at the side-swap tick are ignored.
Evidence for A6 in the raw rows (verify_D1_10_probe.py section A): the +1 scoring row precedes the swap
reassignment rows (team 2 15->12 and team 3 12->15) within the same round.

Compares the first-+1 side with round_end.winner_team_code for ALL 2026 rounds, and specifically for the
105 claim-flagged rows (rounds 27, 33, 39, 45). If A6 is right, the first-+1 side should agree with the
winner in every round, including the flagged ones, which the single-+1 rule could not resolve.

ASSUMPTIONS: A1 team codes 2 = T, 3 = CT; A6 as above. Read-only; no file output.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)
pl.Config.set_fmt_str_lengths(36)
pl.Config.set_tbl_width_chars(200)

ROOT = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def load(ch, cols):
    p = sorted(glob.glob(str(ROOT / f"basic_{ch}*" / "dataframe_*")))
    if not p:
        raise SystemExit(f"no files for {ch}")
    return pl.scan_parquet(p).select(cols).collect().with_columns(
        pl.col("round").cast(pl.Int64), pl.col("match_key").cast(pl.String))


RE = load("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"])
SU = load("score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"])
RS = load("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])

g = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_b"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
        pl.col("t_score").sort_by("tick").last().alias("t_a"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_a")))
g = g.with_columns(
    pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2))
      .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("claim_side"))

inc = SU.filter((pl.col("new_score") - pl.col("old_score")) == 1).sort(["match_key", "round", "tick"])
first = inc.group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("team_code").first().alias("first_side"),
    pl.col("tick").first().alias("first_tick"),
    pl.len().alias("n_inc"))
# tie check: is the earliest +1 row unique by tick within the round?
tie = inc.group_by(["match_key", "round"]).agg(
    (pl.col("tick") == pl.col("tick").min()).sum().alias("n_at_first_tick"),
    pl.col("team_code").sort_by("tick").first().alias("tc_first_check"))

J = (RE.join(g, on=["match_key", "round"], how="left")
       .join(first, on=["match_key", "round"], how="left")
       .join(tie.select(["match_key", "round", "n_at_first_tick"]), on=["match_key", "round"], how="left"))
J = J.with_columns(
    (pl.col("first_side") != pl.col("winner_team_code")).alias("first_dis"),
    (pl.col("claim_side").is_not_null() & (pl.col("claim_side") != pl.col("winner_team_code"))).alias("claim_dis"),
    (pl.col("first_side").is_null()).alias("no_first"))

print("==== verify_D1_10_firstscore (2026) ====")
print("round_end unique rounds:", J.height, " rounds with a +1 score_update row:", J.filter(~pl.col("no_first")).height)
print("rounds with no +1 score_update row at all:", J.filter(pl.col("no_first")).height)
print("rounds where the earliest +1 row is tied on tick with another +1 row:",
      J.filter(pl.col("n_at_first_tick") > 1).height)
print(f"\nfirst-+1 side vs winner_team_code, ALL 2026 rounds: "
      f"disagreements={J.filter(pl.col('first_dis')).height} of {J.filter(~pl.col('no_first')).height} with a first side")

flag = J.filter(pl.col("claim_dis"))
print(f"\nclaim-flagged rows (pc all): {flag.height}")
print("  first-+1 side present:", flag.filter(~pl.col("no_first")).height,
      " first-+1 disagreements:", flag.filter(pl.col("first_dis")).height,
      " first-+1 agrees with winner:", flag.filter((~pl.col("no_first")) & (~pl.col("first_dis"))).height)
print("  by round:", flag.group_by("round").len().sort("round").rows())

print("\nper round, player_count 10 (n, first_dis, first_side_present, claim_dis):")
pc = J.filter(pl.col("player_count") == 10)
per = pc.group_by("round").agg(
    pl.len().alias("n"),
    pl.col("first_dis").sum().alias("first_dis"),
    (~pl.col("no_first")).sum().alias("first_present"),
    pl.col("claim_dis").sum().alias("claim_dis")).sort("round")
for r in per.iter_rows(named=True):
    if r["round"] in (1, 2, 9, 12, 13, 25, 26, 27, 28, 33, 39, 45) or r["first_dis"] or r["claim_dis"]:
        print(f"  round {r['round']:>2}: n={r['n']:>5} first_present={r['first_present']:>5} first_dis={r['first_dis']:>3} claim_dis={r['claim_dis']:>3}")
print("DONE")
