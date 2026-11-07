"""verify_D1_10_probe.py -- follow-up probes for claim D1-10 (swarm2 verifier). Read-only, stdout only.

Section A (2026): raw score_update rows around the flagged OT round 27, to see why the independent
score_update channel gives no single +1 side for those rounds.
Section B (2024): bomb-outcome disagreements by match day, split into round 13 and the other flagged
rounds (9, 25, 28), to reconcile against the claim's 434/605, 0/180 and 41/14,602 figures.

ASSUMPTIONS: as in verify_D1_10_recompute.py (A1 2 = T, 3 = CT; A4 bomb side rule). Day = match_key path
day (csds/YYYY/MM/DD/...). The claim's 14,602 denominator scope is unknown to this script; not reproduced.
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

TOMES = {
    "2026": Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"),
    "2024": Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"),
}


def load(root, ch, cols):
    p = sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))
    if not p:
        return None
    return pl.scan_parquet(p).select(cols).collect().with_columns(
        pl.col("round").cast(pl.Int64), pl.col("match_key").cast(pl.String))


# ---------- Section A: 2026 raw score_update around round 27 ----------
root = TOMES["2026"]
RE = load(root, "round_end", ["match_key", "round", "winner_team_code", "player_count"])
SU = load(root, "score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"])
RS = load(root, "round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
BS = load(root, "bomb_state", ["match_key", "round", "event_type", "tick"])
print("==== Section A: 2026 raw rows, flagged OT round 27 ====")
flag = (RE.filter((pl.col("round") == 27) & (pl.col("player_count") == 10))
          .sort("match_key").head(3).select(["match_key", "winner_team_code"]))
for row in flag.iter_rows(named=True):
    mk, w = row["match_key"], row["winner_team_code"]
    print(f"\nmatch {mk[-40:]}  round_end winner_team_code={w}")
    su_r = SU.filter((pl.col("match_key") == mk) & pl.col("round").is_in([26, 27, 28])).sort(["round", "tick"])
    print("  score_update rows (rounds 26-28):")
    print(su_r.select(["round", "tick", "team_code", "old_score", "new_score"]))
    rs_r = RS.filter((pl.col("match_key") == mk) & (pl.col("round") == 27)).sort("tick")
    # collapse consecutive identical score pairs to show the score path in round 27
    path = rs_r.with_columns(
        ((pl.col("t_score") != pl.col("t_score").shift(1)) | (pl.col("ct_score") != pl.col("ct_score").shift(1))).alias("chg"))
    print("  round_state score path in round 27 (tick, t_score, ct_score) at each change:")
    print(path.filter(pl.col("chg") | pl.col("chg").is_null()).select(["tick", "t_score", "ct_score"]))
    if BS is not None:
        bb = BS.filter((pl.col("match_key") == mk) & (pl.col("round") == 27))
        print("  bomb_state rows in round 27:", bb.group_by("event_type").len().rows())

# how many +1 rows per flagged round, and do their team codes cover both teams
su_inc = (SU.filter((pl.col("new_score") - pl.col("old_score")) == 1)
            .group_by(["match_key", "round"]).agg(
                pl.len().alias("n_inc"),
                pl.col("team_code").n_unique().alias("n_teams")))
m27 = RE.filter((pl.col("round") == 27) & (pl.col("player_count") == 10)).join(
    su_inc, on=["match_key", "round"], how="left")
print("\nround 27, player_count 10: number of +1 score_update rows (n_inc) vs distinct team codes:")
print(m27.group_by(["n_inc", "n_teams"]).len().sort(["n_inc", "n_teams"]).rows())

# ---------- Section B: 2024 bomb disagreements by day ----------
root24 = TOMES["2024"]
RE4 = load(root24, "round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"])
BS4 = load(root24, "bomb_state", ["match_key", "round", "event_type"])
bomb4 = BS4.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_exploded").any().alias("b_exp"),
    (pl.col("event_type") == "bomb_defused").any().alias("b_def"))
bomb4 = bomb4.with_columns(
    pl.when(pl.col("b_exp") & ~pl.col("b_def")).then(pl.lit(2))
      .when(pl.col("b_def") & ~pl.col("b_exp")).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
J4 = RE4.join(bomb4.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")
J4 = J4.filter(pl.col("bomb_side").is_not_null()).with_columns(
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("dis"),
    pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).cast(pl.Int64).alias("day"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round").is_in([9, 25, 28])).then(pl.lit("r9_25_28")).otherwise(pl.lit("other")).alias("grp"))
print("\n==== Section B: 2024 bomb-labelled rounds and disagreements by day and group (all player_count) ====")
print("total bomb-labelled:", J4.height, " disagreements:", J4.filter(pl.col("dis")).height)
print("by group:", J4.group_by("grp").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("grp").rows())
byday = (J4.group_by(["day", "grp"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"))
           .sort(["day", "grp"]))
for r in byday.iter_rows(named=True):
    if r["grp"] == "other" and r["dis"] == 0 and r["n"] == 0:
        continue
    print(f"  day {r['day']:>2} {r['grp']:<9} n={r['n']:>5} dis={r['dis']:>4}")
# claim window split (days 12-25 vs 1-7), other rounds and round 13
win = J4.with_columns(
    pl.when(pl.col("day").is_between(12, 25)).then(pl.lit("02-12..25"))
      .when(pl.col("day").is_between(1, 7)).then(pl.lit("02-01..07"))
      .otherwise(pl.lit("other days")).alias("window"))
print("\nby window x group (n, disagreements):")
for r in win.group_by(["window", "grp"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort(["window", "grp"]).iter_rows(named=True):
    print(f"  {r['window']:<12} {r['grp']:<9} n={r['n']:>5} dis={r['dis']:>4}")
print("DONE")
