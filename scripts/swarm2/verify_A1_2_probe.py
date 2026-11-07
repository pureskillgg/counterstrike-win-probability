# verify A1-2 probe: list channels and schemas for the 2026 tome (read-only).
import os, glob
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
chans = sorted(d for d in os.listdir(T) if os.path.isdir(os.path.join(T, d)))
print("channel folders:", len(chans))
for c in chans:
    print("  ", c)

hdr_files = sorted(glob.glob(os.path.join(T, "header", "dataframe_*")))
print("header files:", hdr_files)
h = pl.scan_parquet(hdr_files).limit(3).collect()
print("header schema:")
for name, dt in pl.scan_parquet(hdr_files).collect_schema().items():
    print("   ", name, dt)
print("header sample row 0:")
print(h.row(0, named=True))

re_files = sorted(glob.glob(os.path.join(T, "basic_round_end.2026-09-18,2026-10-08", "dataframe_*")))
print("round_end files:", len(re_files))
print("round_end schema:")
for name, dt in pl.scan_parquet(re_files).collect_schema().items():
    print("   ", name, dt)
