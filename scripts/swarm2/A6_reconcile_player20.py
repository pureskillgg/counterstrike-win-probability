"""A6 reconciliation: bomb-side mismatch with the full-round filter (player_count == 20), 2024.

Compares with the figures quoted in the task (434 of 605 round-13 rows; 0 of 180 in days 01-07;
41 of 14,602 other rounds) and with the raw counts from A6_label_audit.py.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_reconcile_player20.py
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


RE = load("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
BS = load("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bo = BS.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_defused").any().alias("defused"),
    (pl.col("event_type") == "bomb_exploded").any().alias("exploded"))
day = pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1)
for label, flt in [("all rows", pl.lit(True)), ("player_count == 20", pl.col("player_count") == 20)]:
    j = RE.filter(flt).join(bo, on=["match_key", "round"], how="inner").with_columns(
        day.alias("day"),
        pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rbin"),
        ((pl.col("defused") & (pl.col("winner_team_code") != CT)) |
         (pl.col("exploded") & (pl.col("winner_team_code") != T))).alias("mis"))
    r13 = j.filter(pl.col("rbin") == "r13")
    oth = j.filter(pl.col("rbin") == "other")
    early = r13.filter(pl.col("day").cast(pl.Int32) <= 7)
    late = r13.filter(pl.col("day").cast(pl.Int32) >= 12)
    print(f"[{label}] r13 rows with bomb outcome={r13.height}, mismatches={r13.filter('mis').height if False else int(r13['mis'].sum())}")
    print(f"[{label}] r13 days 01-07: rows={early.height}, mismatches={int(early['mis'].sum())}; "
          f"days 12-25: rows={late.height}, mismatches={int(late['mis'].sum())}")
    print(f"[{label}] other rounds with bomb outcome={oth.height}, mismatches={int(oth['mis'].sum())}")
    print(f"[{label}] day rate r13:", r13.group_by("day").agg(pl.len().alias("n"), pl.col("mis").sum().alias("mis"))
          .sort("day").with_columns((pl.col("mis") / pl.col("n")).round(3).alias("rate")).rows())
