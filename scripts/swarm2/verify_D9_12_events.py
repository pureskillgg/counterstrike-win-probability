"""D9-12 verifier, step 1: event-type vocabularies and per-tome key counts (read-only, projected scans)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024-02-g": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026-10-01_to_08": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}


def scan(base, suf, ch, cols):
    files = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    return pl.scan_parquet(files).select(cols).collect()


for tome, (base, suf) in TOMES.items():
    print(f"== tome {tome}")
    rs = scan(base, suf, "round_state", ["match_key", "round", "event_type"])
    print("   round_state event_type counts:", rs.group_by("event_type").len().sort("len", descending=True).rows())
    bs = scan(base, suf, "bomb_state", ["match_key", "round", "event_type"])
    print("   bomb_state event_type counts:", bs.group_by("event_type").len().sort("len", descending=True).rows())
    re_ = scan(base, suf, "round_end", ["match_key", "round", "tick"])
    rst = scan(base, suf, "round_start", ["match_key", "round", "tick"])
    print(f"   round_end rows={re_.height} unique(match,round)={re_.unique(['match_key', 'round']).height} "
          f"matches={re_['match_key'].n_unique()}")
    print(f"   round_start rows={rst.height} unique(match,round)={rst.unique(['match_key', 'round']).height} "
          f"matches={rst['match_key'].n_unique()}")
    print("   round_start round-label min/max:", rst.select(pl.col('round').min(), pl.col('round').max()).rows(),
          "| round_end round-label min/max:", re_.select(pl.col('round').min(), pl.col('round').max()).rows())
    print("   round_start round-label value counts (lowest 5):",
          rst.group_by("round").len().sort("round").head(5).rows())
    print("   round_end round-label value counts (lowest 5):",
          re_.group_by("round").len().sort("round").head(5).rows())
    pdth = scan(base, suf, "player_death", ["match_key", "round", "tick"])
    print("   player_death rows:", pdth.height)
