"""verify_D9_10 step 13: two checks on the summary numbers.
  (a) the 467 death rows flagged 'disagree' in earlier steps: are they null player_team_code rows (missing side) rather than inverted labels?
  (b) do the 9-halftime matches look like a different format? compare max round, map, final scores and header fields
      between 9-halftime and 13-halftime build-9961 matches (read-only, projected columns, stdout only).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def scan(ch, cols, header=False):
    fd = BASE / "header" if header else BASE / f"basic_{ch}{SUF}"
    lf = pl.scan_parquet(sorted(glob.glob(str(fd / "dataframe_*"))))
    have = set(lf.collect_schema().names())
    missing = [c for c in cols if c not in have]
    if missing:
        print("  missing columns:", missing)
    return lf.select([c for c in cols if c in have]).collect()


H = scan("header", ["key", "build_num", "map_name", "match_type", "number_of_points", "t_starters_score_final",
                    "ct_starters_score_final", "tick_rate", "game_directory"], header=True).rename({"key": "match_key"})
H9961 = H.filter(pl.col("build_num") == 9961)

# (a) death rows with null / non-null player_team_code, and PI team codes null
DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"])
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
j = DEu.join(PIu.rename({"team_code": "it"}), on=["match_key", "round", "player_id"], how="inner").join(
    H9961.select(["match_key"]), on="match_key", how="inner")
print("(a) build 9961 death rows joined to player_info:", j.height)
print("    death player_team_code null:", j.filter(pl.col("player_team_code").is_null()).height,
      "| player_info team_code null:", j.filter(pl.col("it").is_null()).height)
print("    both non-null and equal:", j.filter(pl.col("player_team_code").is_not_null() & pl.col("it").is_not_null()
                                             & (pl.col("player_team_code") == pl.col("it"))).height,
      "| both non-null and different:", j.filter(pl.col("player_team_code").is_not_null() & pl.col("it").is_not_null()
                                                 & (pl.col("player_team_code") != pl.col("it"))).height)
print("    death player_team_code values among non-null:", j.filter(pl.col("player_team_code").is_not_null()).group_by("player_team_code").len().sort("player_team_code").rows())
print("    null death codes by round (top):", j.filter(pl.col("player_team_code").is_null()).group_by("round").len().sort("len", descending=True).head(8).rows())

# (b) format comparison: 9-halftime vs 13-halftime (from first-switch rule of verify_D9_10_12_summary.py)
RE = scan("round_end", ["match_key", "round", "winner_team_code"]).unique(subset=["match_key", "round"], keep="first")
pr = PIu.sort(["match_key", "player_id", "round"]).with_columns(
    pl.col("team_code").shift(1).over(["match_key", "player_id"]).alias("prev_team"),
    pl.col("round").shift(1).over(["match_key", "player_id"]).alias("prev_round"))
pr = pr.filter(pl.col("prev_team").is_not_null() & (pl.col("round") - pl.col("prev_round") == 1))
fc = (pr.with_columns((pl.col("team_code") != pl.col("prev_team")).alias("chg"))
        .group_by(["match_key", "round"]).agg(pl.col("chg").mean().alias("frac")))
half = (fc.filter((pl.col("frac") > 0.5) & pl.col("round").is_in([9, 13])).join(H9961.select("match_key"), on="match_key")
          .group_by("match_key").agg(pl.col("round").min().alias("halftime_r")))
M = H9961.join(half, on="match_key", how="left").with_columns(pl.col("halftime_r").fill_null(-1).alias("ht"))
maxr = RE.join(H9961.select("match_key"), on="match_key").group_by("match_key").agg(pl.col("round").max().alias("max_round"),
                                                                                  pl.len().alias("n_rounds"))
M = M.join(maxr, on="match_key", how="left")
print("\n(b) build 9961 matches by halftime class (-1 = none):")
print(M.group_by("ht").agg(pl.len().alias("matches"), pl.col("max_round").mean().alias("mean_max_round"),
                           pl.col("max_round").min().alias("min_max_round"), pl.col("max_round").max().alias("max_max_round"),
                           pl.col("n_rounds").mean().alias("mean_rounds")).sort("ht"))
print("    max_round distribution for ht=9:", M.filter(pl.col("ht") == 9).group_by("max_round").len().sort("max_round").rows())
print("    max_round distribution for ht=13 (top 12):", M.filter(pl.col("ht") == 13).group_by("max_round").len().sort("max_round").head(12).rows())
print("    map_name by ht:", M.group_by(["ht", "map_name"]).len().sort(["ht", "len"], descending=[False, True]).head(12).rows())
print("    match_type by ht:", M.group_by(["ht", "match_type"]).len().sort("ht").rows())
print("    tick_rate by ht:", M.group_by(["ht", "tick_rate"]).len().sort("ht").rows())
print("    final score sums (t+ct) for ht=9 (top 8):", M.filter(pl.col("ht") == 9).with_columns(
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("s")).group_by("s").len().sort("len", descending=True).head(8).rows())
print("    final score sums (t+ct) for ht=13 (top 8):", M.filter(pl.col("ht") == 13).with_columns(
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("s")).group_by("s").len().sort("len", descending=True).head(8).rows())
