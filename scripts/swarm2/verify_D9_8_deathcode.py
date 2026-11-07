"""verify D9-8, part 7: do player_death team codes agree with player_info team codes for the same
player and round? (2024 tome, read-only.) Needed before trusting verify_D9_8_elim.py on round 13.
Agreement by build x round bin, victims only (player_death.player_team_code vs player_info.team_code).
ASSUMPTIONS: A1 codes 2 = T, 3 = CT. Output is printed; no files are written under E:.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files_of(name):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    return sorted(glob.glob(str(fd / "dataframe_*")))


def lz(name, cols):
    lf = pl.scan_parquet(files_of(name))
    have = lf.collect_schema().names()
    miss = [c for c in cols if c not in have]
    if miss:
        raise SystemExit(f"{name} missing columns {miss}")
    return lf.select(cols)


H = lz("header", ["key", "build_num"]).collect().rename({"key": "match_key"})
DE = lz("player_death", ["match_key", "round", "player_id", "player_team_code"]).collect()
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
PI = lz("player_info", ["match_key", "round", "player_id", "team_code"]).collect()
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
J = DEu.join(PIu, on=["match_key", "round", "player_id"], how="inner").join(
    H.select(["match_key", "build_num"]), on="match_key", how="left")
J = J.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
      .when(pl.col("round") == 9).then(pl.lit("r9")).when(pl.col("round") == 28).then(pl.lit("r28"))
      .otherwise(pl.lit("other")).alias("rbin"),
    (pl.col("player_team_code") == pl.col("team_code")).alias("agree"),
)
print("victim rows matched to player_info:", J.height, "of", DEu.height, "distinct victim rows")
print(J.group_by(["build_num", "rbin"]).agg(pl.len().alias("n"), pl.col("agree").sum().alias("agree"),
                                           pl.col("agree").mean().alias("rate")).sort(["build_num", "rbin"]))
print("\nDONE")
