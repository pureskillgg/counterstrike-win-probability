"""D1-9 verification, part 4: is the 2024 player_count regime set per match? (read-only)

For each 2024 match, classify rounds by spawn-based count n_sp:
  pc == 20          -> 'constant 20'
  pc == 2 * n_sp    -> 'doubled count'
  otherwise         -> 'other'
Then classify matches by which rule covers all their rounds with spawn data.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(90)
pl.Config.set_tbl_width_chars(200)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
S24 = ".2023-12-28,2024-02-25"


def lf(base, suf, ch):
    fs = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no files for {ch} under {base}")
    return pl.scan_parquet(fs)


RE = lf(T24, S24, "round_end").select("match_key", "round", "player_count").collect()
RE = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
SP = lf(T24, S24, "player_spawn").select("match_key", "round", "player_id").collect()
SR = SP.group_by(["match_key", "round"]).agg(pl.col("player_id").n_unique().alias("n_sp"))
J = RE.join(SR, on=["match_key", "round"], how="left").filter(pl.col("n_sp").is_not_null())
J = J.with_columns(
    is20=pl.col("player_count") == 20,
    is2n=pl.col("player_count") == 2 * pl.col("n_sp"),
)
M = J.group_by("match_key").agg(
    pl.len().alias("rounds"),
    pl.col("is20").sum().alias("n20"),
    pl.col("is2n").sum().alias("n2n"),
    pl.col("n_sp").max().alias("max_nsp"),
    pl.col("n_sp").min().alias("min_nsp"),
)
M = M.with_columns(
    cls=pl.when(pl.col("n20") == pl.col("rounds")).then(pl.lit("all_20"))
        .when(pl.col("n2n") == pl.col("rounds")).then(pl.lit("all_2n"))
        .otherwise(pl.lit("mixed")),
)
print("2024 matches with spawn data:", M.height)
print(M.group_by("cls").len().sort("cls"))
print("\nRounds by match class:")
print(J.join(M.select(["match_key", "cls"]), on="match_key").group_by("cls")
      .agg(pl.len().alias("rounds"), pl.col("is20").sum().alias("pc20"), pl.col("is2n").sum().alias("pc_2n"))
      .sort("cls"))
print("\nall_20 matches: distribution of max_nsp (roster size seen in spawns):")
print(M.filter(pl.col("cls") == "all_20").group_by("max_nsp").len().sort("max_nsp"))
print("\nall_2n matches: distribution of max_nsp:")
print(M.filter(pl.col("cls") == "all_2n").group_by("max_nsp").len().sort("max_nsp"))
print("\nmixed matches: distribution of max_nsp:")
print(M.filter(pl.col("cls") == "mixed").group_by("max_nsp").len().sort("max_nsp"))
mixed_share = M.filter(pl.col("cls") == "mixed").with_columns(
    share20=pl.col("n20") / pl.col("rounds"))
print("\nmixed matches: share of rounds at 20 (quantiles):")
print(mixed_share["share20"].describe())
print("\nDONE")
