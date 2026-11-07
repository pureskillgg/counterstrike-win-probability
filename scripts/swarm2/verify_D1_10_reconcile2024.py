"""verify_D1_10_reconcile2024.py -- reconcile the claim's 2024 bomb figures (read-only, stdout only).

Claim figures to match (from the claim text): round 13 with a bomb outcome in matches dated 2024-02-12..25:
434 of 605 mismatch; 0 of 180 in 2024-02-01..07; other rounds 41 of 14,602.
This script tabulates bomb-labelled rounds and mismatches for round 13, rounds 9/25/28, and all other rounds,
by day window, for player_count 20 (the claim's e2 check used pc20 per the D1 context) and for all player counts.

ASSUMPTIONS: A4 bomb side rule (exploded only -> T=2, defused only -> CT=3); day from match_key path;
pc20 filter as a hypothesis for the claim's denominators, not confirmed.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")


def tome(ch):
    p = sorted(glob.glob(str(ROOT / f"basic_{ch}*" / "dataframe_*")))
    return pl.scan_parquet(p)


RE = (tome("round_end").select(["match_key", "round", "winner_team_code", "player_count"]).collect()
      .unique(["match_key", "round"]))
BS = tome("bomb_state").select(["match_key", "round", "event_type"]).collect()
bomb = BS.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_exploded").any().alias("b_exp"),
    (pl.col("event_type") == "bomb_defused").any().alias("b_def"))
bomb = bomb.with_columns(
    pl.when(pl.col("b_exp") & ~pl.col("b_def")).then(pl.lit(2))
      .when(pl.col("b_def") & ~pl.col("b_exp")).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
J = RE.join(bomb.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="inner")
J = J.filter(pl.col("bomb_side").is_not_null())  # keep only rounds with exactly one bomb outcome (A4)
J = J.with_columns(
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("dis"),
    pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).cast(pl.Int64).alias("day"))
J = J.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round").is_in([9, 25, 28])).then(pl.lit("r9_25_28"))
      .otherwise(pl.lit("other")).alias("grp"),
    pl.when(pl.col("day").is_between(12, 25)).then(pl.lit("02-12..25"))
      .when(pl.col("day").is_between(1, 7)).then(pl.lit("02-01..07"))
      .otherwise(pl.lit("other days")).alias("window"))
print("2024 tome, bomb-labelled rounds (all pc):", J.height, " mismatches:", J.filter(pl.col("dis")).height)
print("player_count distribution among bomb-labelled rounds (top 5):",
      J.group_by("player_count").len().sort("len", descending=True).head(5).rows())

for label, sub in [("player_count == 20", J.filter(pl.col("player_count") == 20)), ("all player_count", J)]:
    print(f"\n--- {label} ---")
    t = sub.group_by(["window", "grp"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("mismatch")).sort(["window", "grp"])
    for r in t.iter_rows(named=True):
        print(f"  {r['window']:<12} {r['grp']:<9} n={r['n']:>6} mismatch={r['mismatch']:>4}")
    r13 = sub.filter(pl.col("grp") == "r13")
    print(f"  round 13 total: n={r13.height} mismatch={r13.filter(pl.col('dis')).height}")
    oth = sub.filter(pl.col("grp") != "r13")
    print(f"  rounds other than 13: n={oth.height} mismatch={oth.filter(pl.col('dis')).height}")
    pr = (oth.group_by("round").agg(pl.len().alias("n"), pl.col("dis").sum().alias("mismatch"))
            .sort("round").filter(pl.col("mismatch") > 0))
    print("  per-round mismatches (other than 13):", pr.rows())
    pr_all = (sub.group_by("round").agg(pl.len().alias("n"), pl.col("dis").sum().alias("mismatch"))
                .sort("round").filter(pl.col("round").is_in([9, 25, 28])))
    print("  rounds 9/25/28 (n, mismatch):", pr_all.rows())
print("DONE")
