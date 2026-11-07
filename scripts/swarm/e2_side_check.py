"""Check that round_end.winner_team_code uses side codes (2 = T, 3 = CT) for E2.

Uses bomb events as ground truth: a defused bomb is a CT win, an exploded bomb is a T win.
Also checks that a planted bomb is never followed by a T loss without explosion/defuse is not checked here.
Run: pixi run python -I -X utf8 scripts\\swarm\\e2_side_check.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
ERAS = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}


def read(base, suffix, ch, cols):
    fd = base / f"basic_{ch}{suffix}"
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


for era, (base, suffix) in ERAS.items():
    RE = read(base, suffix, "round_end", ["match_key", "round", "winner_team_code"])
    BS = read(base, suffix, "bomb_state", ["match_key", "round", "event_type"])
    ev = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
          .group_by(["match_key", "round"])
          .agg(pl.col("event_type").unique().alias("evs")))
    j = RE.join(ev, on=["match_key", "round"], how="inner").with_columns(
        pl.col("evs").list.contains("bomb_defused").alias("defused"),
        pl.col("evs").list.contains("bomb_exploded").alias("exploded"),
    )
    print(f"\n### {era}: winner codes overall {RE['winner_team_code'].unique().sort().to_list()}")
    print(f"rounds with defuse/explode rows: {j.height}")
    print(j.group_by(["defused", "exploded", "winner_team_code"]).len().sort(["defused", "exploded", "winner_team_code"]))
