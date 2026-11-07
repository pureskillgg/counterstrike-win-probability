"""Feasibility: schema check for grenade / molotov / hurt / blind / bomb channels (read-only).

Prints column names, dtypes, row counts, distinct event_type values (where present),
and null share per position column. Run: pixi run python -I -X utf8 scripts\\eda\\feas_grenade_schema.py
"""
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"

CHANNELS = ["round_end", "round_state", "player_death", "player_hurt", "player_blind",
            "grenade_state", "molotov_state", "molotov_fire",
            "bomb_state", "bomb_action", "bomb_defuse"]
POS_COLS_HINT = ("x", "y", "z", "pos", "position", "origin", "location")


def read(ch):
    folder = TOME / f"basic_{ch}{SUFFIX}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    frames = [pl.read_parquet(f) for f in files]
    return pl.concat(frames, how="vertical_relaxed"), len(files)


def main():
    for ch in CHANNELS:
        df, nfiles = read(ch)
        print(f"\n=== {ch}: rows={df.height} files={nfiles}")
        print("columns:")
        for name, dtype in df.schema.items():
            print(f"  {name}: {dtype}")
        if "event_type" in df.columns:
            vc = df.group_by("event_type").len().sort("len", descending=True)
            print("event_type values:")
            for row in vc.iter_rows():
                print(f"  {row[0]!r}: {row[1]}")
        pos_cols = [c for c in df.columns if c.lower() in
                    ("x", "y", "z", "x_pos", "y_pos", "z_pos") or c.lower().endswith(("_x", "_y", "_z"))
                    or "position" in c.lower()]
        if pos_cols:
            print("null share per position column:")
            for c in pos_cols:
                n_null = df.get_column(c).null_count()
                print(f"  {c}: {n_null}/{df.height} = {n_null / max(df.height, 1):.4f}")
        else:
            print("no position-like columns found")
        for c in ("tick", "round", "match_key"):
            if c in df.columns:
                print(f"  {c} null share: {df.get_column(c).null_count() / max(df.height, 1):.4f}")


if __name__ == "__main__":
    main()
