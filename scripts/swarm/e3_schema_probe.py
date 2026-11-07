"""E3 schema probe: column lists, event types and key counts for the channels the kill
analysis needs, on both G: tomes. Reads G: tome pages only (small aggregates)."""
import glob
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}
CH = ["header", "round_start", "round_end", "round_state", "player_death", "player_info",
      "bomb_state", "player_disconnect", "bot_takeover", "team_change"]


def folder(base, suffix, ch):
    return base / ("header" if ch == "header" else f"basic_{ch}{suffix}")


for era, (base, suffix) in TOMES.items():
    print(f"===== {era}")
    for ch in CH:
        fd = folder(base, suffix, ch)
        if not fd.exists():
            print(f"  {ch}: FOLDER MISSING")
            continue
        files = sorted(glob.glob(str(fd / "dataframe_*")))
        if not files:
            print(f"  {ch}: no dataframe files")
            continue
        names = pq.read_schema(files[0]).names
        ks = sorted(glob.glob(str(fd / "keyset_*")))
        print(f"  {ch}: files={len(files)} cols={names}")
        if ks:
            keys = pl.concat([pl.read_parquet(k) for k in ks])
            print(f"    keyset columns={keys.columns} n_keys={keys.height}")
        if ch == "round_state":
            df = pl.concat([pl.read_parquet(f, columns=["event_type"]) for f in files], how="vertical_relaxed")
            print("    event_type:", df.group_by("event_type").len().sort("len", descending=True).rows())
        if ch == "round_end":
            df = pl.concat([pl.read_parquet(f, columns=["winner_team_code"]) for f in files], how="vertical_relaxed")
            print("    winner_team_code:", df.group_by("winner_team_code").len().rows())
        if ch == "player_info":
            df = pl.concat([pl.read_parquet(f, columns=["team_code"]) for f in files], how="vertical_relaxed")
            print("    team_code:", df.group_by("team_code").len().rows())
        if ch == "header":
            df = pl.concat([pl.read_parquet(f, columns=["tick_rate"]) for f in files], how="vertical_relaxed")
            print("    tick_rate:", df.group_by("tick_rate").len().rows())
