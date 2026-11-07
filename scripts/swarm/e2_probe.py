"""E2 probe: schemas and row counts for the channels the map profile needs, both tomes. Reads G: only."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}
CH = ["header", "round_start", "round_end", "round_state", "bomb_state", "player_death"]


def folder(base, suffix, ch):
    return base / ("header" if ch == "header" else f"basic_{ch}{suffix}")


for era, (base, suffix) in TOMES.items():
    print(f"===== {era} base={base}")
    print("folders:", sorted(p.name for p in base.iterdir())[:3], "... total", len(list(base.iterdir())))
    for ch in CH:
        fd = folder(base, suffix, ch)
        files = sorted(glob.glob(str(fd / "dataframe_*")))
        if not files:
            print(f"  {ch}: NO FILES in {fd}")
            continue
        df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
        print(f"  {ch}: files={len(files)} rows={df.height} cols={df.width}")
        print("    schema:", {k: str(v) for k, v in df.schema.items()})
        if ch in ("round_state", "bomb_state", "round_end"):
            col = "event_type" if "event_type" in df.columns else None
            if col:
                print("    event_type counts:", df.group_by(col).len().sort("len", descending=True).head(25).rows())
        if ch == "header":
            print(df.select([c for c in ["key", "map_name", "is_wingman", "tick_rate", "tick_save_rate", "max_rounds", "match_type", "platform", "match_date"] if c in df.columns]).head(3))
            print("    map_name counts:", df.group_by("map_name").len().sort("len", descending=True).rows())
            if "tick_rate" in df.columns:
                print("    tick_rate counts:", df.group_by("tick_rate").len().rows())
            if "is_wingman" in df.columns:
                print("    is_wingman counts:", df.group_by("is_wingman").len().rows())
        if ch == "player_death":
            print(df.head(3))
