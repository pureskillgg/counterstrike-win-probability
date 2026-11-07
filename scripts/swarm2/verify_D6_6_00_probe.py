"""D6-6 verifier step 00: schema, parquet writer metadata and match-day layout (read-only).

Checks whether the 2024 tome or the raw E: tail-recovered copy shows any build/export difference between
match days 01-07 and 12-25. Reads footers and headers only.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_00_probe.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
BASE26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
RAW24 = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")

print("== A. 2024 header schema (tome) ==")
hf = sorted(glob.glob(str(BASE24 / "header" / "dataframe_*")))
print("header files:", len(hf))
md = pq.ParquetFile(hf[0]).metadata
print("created_by:", md.created_by)
print("format_version:", md.format_version, "row_groups:", md.num_row_groups, "rows:", md.num_rows)
sch = pq.ParquetFile(hf[0]).schema_arrow
print("columns:", sch.names)
kv = sch.metadata or {}
print("schema metadata keys:", [k.decode(errors='replace') for k in kv.keys()][:20])

print("\n== B. 2024 header per-day counts and map (tome) ==")
H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "map_name"]) for f in hf], how="vertical_relaxed")
H = H.with_columns(pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
print("header rows:", H.height, "unique keys:", H["key"].n_unique())
print(H.group_by("day").len().sort("day"))
print("match_date sample:", H.select("match_date").head(3).to_series().to_list())

print("\n== C. Raw E: tail-recovered 2024/02 match counts per day (folder count only) ==")
day_counts = {}
for d in sorted(p.name for p in RAW24.iterdir() if p.is_dir()):
    day_counts[d] = sum(1 for q in (RAW24 / d).iterdir() if q.is_dir())
print(day_counts)
print("total raw match folders:", sum(day_counts.values()))

print("\n== D. Raw channel files: writer metadata per day (one sample match per day) ==")
for d in sorted(day_counts):
    dd = RAW24 / d
    mids = [q for q in dd.iterdir() if q.is_dir()]
    if not mids:
        print(d, "no matches")
        continue
    m = mids[0]
    chans = sorted(p.name for p in m.iterdir())
    rep = []
    for ch in ("round_end", "round_state", "bomb_state"):
        p = m / ch
        if p.exists():
            fmd = pq.ParquetFile(str(p)).metadata
            rep.append((ch, fmd.created_by, fmd.num_rows, len(pq.ParquetFile(str(p)).schema_arrow.names)))
    print(f"day {d} sample {m.name[:12]} channels={len(chans)} :: {rep}")

print("\n== E. Tome 2024 round_end schema vs 2026 round_end schema ==")
re24 = sorted(glob.glob(str(sorted(BASE24.glob("basic_round_end.*"))[0] / "dataframe_*")))
re26 = sorted(glob.glob(str(sorted(BASE26.glob("basic_round_end.*"))[0] / "dataframe_*")))
for lab, fl in (("2024", re24), ("2026", re26)):
    m0 = pq.ParquetFile(fl[0]).metadata
    print(lab, "created_by:", m0.created_by, "cols:", pq.ParquetFile(fl[0]).schema_arrow.names)
