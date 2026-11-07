r"""Verifier for D4-5, step B: match length and header score totals, 2024, by day group.

For each 2024 match: number of rounds seen in round_end (distinct round numbers), max round number,
header final score sum (t_starters_score_final + ct_starters_score_final), and whether the round numbers
are exactly 1..n. Tests the D4 assumption that every 2024 match is a 24-round (MR12) regulation match
(swap after round 12, no swap after round 13).

Reads round_end with projected columns (match_key, round, tick) only. Memory: small.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_b_lengths.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(14)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final"]).with_columns(
    pl.col("key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("fsum"))
H = H.with_columns(pl.when(pl.col("day") == "12").then(pl.lit("d12"))
                   .when(pl.col("day").is_in(["01", "02", "03", "04", "05", "06", "07"])).then(pl.lit("d01-07"))
                   .otherwise(pl.lit("d13-25")).alias("grp"))

re_all = read("round_end", ["match_key", "round", "tick", "winner_team_code"])
print(f"round_end rows (2024): {re_all.height}")
per_match = re_all.group_by("match_key").agg(
    pl.len().alias("n_rows"),
    pl.col("round").n_unique().alias("n_distinct"),
    pl.col("round").max().alias("max_round"),
    pl.col("round").min().alias("min_round"),
)
per_match = per_match.join(H.select(["key", "fsum", "grp", "day"]), left_on="match_key", right_on="key", how="left")
print("matches with round_end:", per_match.height, "; header join missing:", per_match.filter(pl.col("fsum").is_null()).height)
print("\nduplicate round_end rows per match (n_rows - n_distinct > 0), by group:")
print(per_match.with_columns((pl.col("n_rows") - pl.col("n_distinct")).alias("dups")).group_by("grp").agg(
    pl.len().alias("matches"), (pl.col("dups") > 0).sum().alias("matches_with_dups"),
    pl.col("dups").sum().alias("dup_rows")).sort("grp"))

print("\nmax_round distribution by group:")
print(per_match.group_by(["grp", "max_round"]).len().sort(["grp", "max_round"]))

print("\nrounds-exactly-1..n (max_round == n_distinct and min_round == 1), by group:")
print(per_match.with_columns(((pl.col("max_round") == pl.col("n_distinct")) & (pl.col("min_round") == 1)).alias("contig"))
      .group_by(["grp", "contig"]).len().sort(["grp", "contig"]))

print("\nheader fsum vs n_distinct (complete regulation check: fsum == n_distinct), by group:")
pm = per_match.with_columns((pl.col("fsum") == pl.col("n_distinct")).alias("fsum_eq_n"))
print(pm.group_by(["grp", "fsum_eq_n"]).len().sort(["grp", "fsum_eq_n"]))

print("\nfsum distribution by group (header starters' final-score sum):")
print(H.group_by(["grp", "fsum"]).len().sort(["grp", "fsum"]))

print("\nmatches with n_distinct > 24 (would violate MR12 assumption), by group:")
print(pm.filter(pl.col("n_distinct") > 24).group_by("grp").len().sort("grp"))
print("\nmatches with fsum > 24, by group:")
print(H.filter(pl.col("fsum") > 24).group_by("grp").len().sort("grp"))
