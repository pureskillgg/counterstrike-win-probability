"""A6 probe: bomb-ended rounds coded 1/7 vs 8/9, by match-day group, 2024 (raw and fingerprint-deduped).

Groups: days 01-07 (before the 2024-02-12 changelog entry) and days 12-25.
Also the same split with the winner-side check (winner == bomb side) and the score-derived side.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_window_bomb_split.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
CT, T = 3, 2


def load(ch, cols):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return pl.scan_parquet(sorted(glob.glob(str(fd / "dataframe_*")))).select(cols).collect()


RE = load("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).unique(
    ["match_key", "round"], keep="first")
BS = load("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bo = BS.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_defused").any().alias("defused"),
    (pl.col("event_type") == "bomb_exploded").any().alias("exploded"))
j = RE.join(bo, on=["match_key", "round"], how="inner").with_columns(
    pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"),
    pl.when(pl.col("defused")).then(pl.lit("defused")).otherwise(pl.lit("exploded")).alias("outcome"))
j = j.with_columns(
    pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01_07")).otherwise(pl.lit("d12_25")).alias("grp"))
j = j.with_columns(
    pl.col("win_reason_code").is_in([1, 7]).alias("code_event"),
    ((pl.col("defused") & (pl.col("winner_team_code") != CT)) |
     (pl.col("exploded") & (pl.col("winner_team_code") != T))).alias("side_mis"))
print("== raw: bomb-ended rounds, group x outcome x code")
print(j.group_by(["grp", "outcome", "win_reason_code"]).len().sort(["grp", "outcome", "win_reason_code"]))
print("\n== raw: share coded 1/7 (event reason) and side mismatch, by group")
print(j.group_by(["grp", "outcome"]).agg(pl.len().alias("n"),
                                         pl.col("code_event").mean().round(4).alias("share_event_code"),
                                         pl.col("side_mis").sum().alias("side_mis")).sort(["grp", "outcome"]))
print("\n== raw: round_end tick offset to bomb event (bomb tick - round_end tick), by group, code family")
BSt = load("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("btick"))
jt = RE.join(BSt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("btick") - pl.col("tick")).alias("off"),
    pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"))
jt = jt.with_columns(
    pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01_07")).otherwise(pl.lit("d12_25")).alias("grp"))
print(jt.group_by(["grp", "win_reason_code"]).agg(pl.len().alias("n"), (pl.col("off") == 0).mean().round(4).alias("share_off0"),
                                                   pl.col("off").median().alias("median_off")).sort(["grp", "win_reason_code"]))
