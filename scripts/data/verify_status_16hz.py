"""Completeness check for the 16-ticks-per-second status store (data/status_16hz/<day>.parquet).

For each upload day in the extraction grid (data/combined/tick16_extract.parquet):
  - the day file must exist;
  - every grid match should have status rows (missing matches are reported; an error only if a whole day is absent);
  - every (match_key, round, tick) in the file must be a grid row (extra keys are an error: the file would describe ticks
    the extraction never asked for);
  - grid ticks with no status rows are counted, not treated as errors: a side with nobody alive at a tick has no rows.

  ...python.exe -I -X utf8 scripts\\data\\verify_status_16hz.py
"""
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
GRID = os.path.join(ROOT, "data", "combined", "tick16_extract.parquet")
STORE = os.path.join(ROOT, "data", "status_16hz")
KEY = ["match_key", "round", "tick"]


def main():
    grid = pl.read_parquet(GRID, columns=KEY + ["upload_day"])
    days = sorted(grid["upload_day"].unique().to_list())
    errors = 0
    total_missing_mk = 0
    print(f"grid: {grid.height:,} rows, {grid['match_key'].n_unique():,} matches, {len(days)} upload days", flush=True)
    for day in days:
        g = grid.filter(pl.col("upload_day") == day).select(KEY)
        f = os.path.join(STORE, day.replace("/", "-") + ".parquet")
        if not os.path.exists(f):
            print(f"{day}: MISSING FILE", flush=True)
            errors += 1
            continue
        out = pl.read_parquet(f, columns=KEY)
        rows = pl.scan_parquet(f).select(pl.len()).collect().item()
        g_mk = set(g["match_key"].unique().to_list())
        o_mk = set(out["match_key"].unique().to_list())
        miss_mk = len(g_mk - o_mk)
        extra_mk = len(o_mk - g_mk)
        g_keys = g.unique()
        o_keys = out.unique()
        extra_keys = o_keys.join(g_keys, on=KEY, how="anti").height
        grid_no_rows = g_keys.join(o_keys, on=KEY, how="anti").height
        total_missing_mk += miss_mk
        flag = "" if (extra_mk == 0 and extra_keys == 0) else "  <-- EXTRA"
        if extra_mk or extra_keys:
            errors += 1
        print(f"{day}: grid matches {len(g_mk)} | file matches {len(o_mk)} | missing {miss_mk} | extra matches {extra_mk} | "
              f"rows {rows:,} | extra keys {extra_keys} | grid (round,tick) with no rows {grid_no_rows:,}{flag}", flush=True)
        del g, out, g_keys, o_keys
    print(f"done: {len(days)} days, errors {errors}, matches missing from files {total_missing_mk}", flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
