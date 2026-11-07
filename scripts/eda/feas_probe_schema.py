"""Feasibility probe (Task A): print the real schema of every tome channel and the sample files.

Read-only. Prints column names and dtypes, file counts and row counts. No values are modified.
Run from the repo root:
  pixi run python -I -X utf8 scripts\\eda\\feas_probe_schema.py
"""

import glob
import os
import sys

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SAMPLE = os.path.join(ROOT, "data", "state_sample")


def show(label, folder_glob):
    files = sorted(glob.glob(os.path.join(folder_glob, "dataframe_*")))
    if not files:
        print(f"{label}: NO dataframe files under {folder_glob}")
        return
    total = 0
    schema = None
    for f in files:
        md = pq.ParquetFile(f).metadata
        total += md.num_rows
        if schema is None:
            schema = pq.ParquetFile(f).schema_arrow
    print(f"{label}: files={len(files)} rows={total}")
    print("   " + ", ".join(f"{n}:{t}" for n, t in zip(schema.names, schema.types)))


def main():
    print("== tome channels (folder list) ==")
    for name in sorted(os.listdir(TOME)):
        print("  " + name)
    print("\n== tome channel schemas ==")
    for name in sorted(os.listdir(TOME)):
        if name == "header":
            continue
        show(name, os.path.join(TOME, name))
    print("\n== header ==")
    show("header", os.path.join(TOME, "header"))
    print("\n== sample files ==")
    for f in ["rounds.parquet", "players_1hz.parquet", "positions_snapshots.parquet"]:
        p = os.path.join(SAMPLE, f)
        if not os.path.exists(p):
            print(f"{f}: missing")
            continue
        pf = pq.ParquetFile(p)
        print(f"{f}: rows={pf.metadata.num_rows}")
        s = pf.schema_arrow
        print("   " + ", ".join(f"{n}:{t}" for n, t in zip(s.names, s.types)))


if __name__ == "__main__":
    main()
