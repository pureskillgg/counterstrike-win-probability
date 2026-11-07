"""verify_D9_10 step 14: is the 9-halftime group an 8-round-half format, and does build 9842 have the same format without inversion?
  - map distribution of 9-halftime matches (full list, both builds)
  - build 9842 9-halftime matches: count, and bomb-round winner check at round 9 (control for the build-specific claim)
  - final score sums for 9-halftime matches by build
Read-only on the tome; projected columns; stdout only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def scan(ch, cols, header=False):
    fd = BASE / "header" if header else BASE / f"basic_{ch}{SUF}"
    lf = pl.scan_parquet(sorted(glob.glob(str(fd / "dataframe_*"))))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in cols if c in have]).collect()


H = scan("header", ["key", "build_num", "map_name", "t_starters_score_final", "ct_starters_score_final"], header=True).rename({"key": "match_key"})
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
pr = PIu.sort(["match_key", "player_id", "round"]).with_columns(
    pl.col("team_code").shift(1).over(["match_key", "player_id"]).alias("prev_team"),
    pl.col("round").shift(1).over(["match_key", "player_id"]).alias("prev_round"))
pr = pr.filter(pl.col("prev_team").is_not_null() & (pl.col("round") - pl.col("prev_round") == 1))
fc = (pr.with_columns((pl.col("team_code") != pl.col("prev_team")).alias("chg"))
        .group_by(["match_key", "round"]).agg(pl.col("chg").mean().alias("frac")))
half = (fc.filter((pl.col("frac") > 0.5) & pl.col("round").is_in([9, 13])).group_by("match_key")
          .agg(pl.col("round").min().alias("halftime_r")))
M = H.join(half, on="match_key", how="left").with_columns(pl.col("halftime_r").fill_null(-1).alias("ht"))
M = M.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("score_sum"))
print("matches by build and halftime class:", M.group_by(["build_num", "ht"]).len().sort(["build_num", "ht"]).rows())
print("\n9-halftime matches: map_name counts (all) by build:")
print(M.filter(pl.col("ht") == 9).group_by(["build_num", "map_name"]).len().sort(["build_num", "len"], descending=[False, True]).rows())
print("\n13-halftime matches: map_name counts (top 12) by build:")
print(M.filter(pl.col("ht") == 13).group_by(["build_num", "map_name"]).len().sort(["build_num", "len"], descending=[False, True]).head(12).rows())
print("\nfinal score sum (t+ct starters) for 9-halftime matches by build (top 8):")
print(M.filter(pl.col("ht") == 9).group_by(["build_num", "score_sum"]).len().sort(["build_num", "len"], descending=[False, True]).group_by("build_num").head(8).sort(["build_num", "len"], descending=[False, True]).rows())

# bomb-round winner check at round 9 in 9-halftime matches of build 9842 (control)
RE = scan("round_end", ["match_key", "round", "winner_team_code"]).unique(subset=["match_key", "round"], keep="first")
BS = scan("bomb_state", ["match_key", "round", "event_type"])
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"])
          .agg((pl.col("event_type") == "bomb_defused").any().alias("defused")))
X = RE.join(outc, on=["match_key", "round"], how="inner").join(M.select(["match_key", "build_num", "ht"]), on="match_key", how="left")
X = X.with_columns(pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
X = X.with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
print("\nbomb rounds at round 9 by build and halftime class (n, mismatches):")
print(X.filter(pl.col("round") == 9).group_by(["build_num", "ht"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["build_num", "ht"]).rows())
print("bomb rounds at round 9 in 9-halftime matches, build 9842: n =",
      X.filter((pl.col("round") == 9) & (pl.col("build_num") == 9842) & (pl.col("ht") == 9)).height)
