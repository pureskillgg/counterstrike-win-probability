"""M2 probe: 2024 player_info team codes and why player_count==20 rounds fail the 5v5 check."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30)
ROOT = Path("G:/pureskill/csds-cs2")
base = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
suf = ".2023-12-28,2024-02-25"


def read(ch, cols):
    fd = base / f"basic_{ch}{suf}"
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(fd / "dataframe_*")))],
                     how="vertical_relaxed")


re_ = read("round_end", ["match_key", "round", "player_count", "winner_team_code"])
pi = read("player_info", ["match_key", "round", "player_id", "team_code"])
print("player_info team_code values:", pi.group_by("team_code").agg(pl.len()).sort("team_code").rows())
print("player_info rows per (match, round, player_id) dup count:",
      pi.group_by(["match_key", "round", "player_id"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1).height)
per = pi.group_by(["match_key", "round"]).agg(
    pl.col("player_id").n_unique().alias("np"),
    pl.len().alias("rows"),
    (pl.col("team_code") == 2).sum().alias("nt"),
    (pl.col("team_code") == 3).sum().alias("nct"),
    pl.col("team_code").n_unique().alias("nteams"))
j = re_.filter(pl.col("player_count") == 20).join(per, on=["match_key", "round"], how="left")
j = j.with_columns(pl.when(pl.col("np").is_null()).then(pl.lit("no_pi"))
                   .when((pl.col("np") == 10) & (pl.col("nt") == 5) & (pl.col("nct") == 5)).then(pl.lit("ok"))
                   .otherwise(pl.lit("other")).alias("cls"))
print("player_count 20 rounds by class:", j.group_by("cls").agg(pl.len()).rows())
bad = j.filter(pl.col("cls") == "other")
print("other: np distribution:", bad.group_by("np").agg(pl.len()).sort("np").head(15).rows())
print("other: (nt, nct) distribution:", bad.group_by(["nt", "nct"]).agg(pl.len()).sort("len", descending=True).head(12).rows())
print("other: rows distribution:", bad.group_by("rows").agg(pl.len()).sort("len", descending=True).head(8).rows())
# per-match: are the 'other' rounds concentrated in a few matches?
mm = bad.group_by("match_key").agg(pl.len().alias("n")).sort("n", descending=True)
print("other rounds: matches affected", mm.height, "top 8 counts:", mm.head(8).rows())
ok_m = j.filter(pl.col("cls") == "ok").group_by("match_key").agg(pl.len().alias("n"))
print("matches with any 'other' round:", mm.height, " matches with any ok round:", ok_m.height)
# sample example of other
ex = bad.head(5)
print(ex.rows())
