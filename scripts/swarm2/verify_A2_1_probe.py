# verify_A2_1 probe: schemas, header key format, page counts for the 2024 tome. Schema-only + small samples.
import glob
import json
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(120)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"


def chan_dir(ch):
    return os.path.join(BASE, f"basic_{ch}{SUF}")


def files_of(ch):
    return sorted(glob.glob(os.path.join(chan_dir(ch), "dataframe_*")))


print("== header files:", files_of_hdr := sorted(glob.glob(os.path.join(BASE, "header", "dataframe_*"))), flush=True)
hs = pl.scan_parquet(files_of_hdr[0])
print("== header schema:")
for k, v in hs.collect_schema().items():
    print("   ", k, v)
h = hs.head(3).collect()
print("== header head (selected cols):")
print(h.select([c for c in ["key", "match_id", "match_date", "map_name", "build_num", "server_name", "tick_rate", "tickrate"] if c in h.columns]))
print("== header all columns sample row 0:")
for k, v in h.row(0, named=True).items():
    print("   ", k, repr(v)[:160])

for ch in ["round_end", "round_state", "bomb_state", "bomb_action", "bomb_defuse"]:
    fs = files_of(ch)
    print(f"== {ch}: {len(fs)} dataframe files; meta pages:", len(json.load(open(os.path.join(chan_dir(ch), "tome"), encoding="utf8"))["pages"]), flush=True)
    sch = pl.scan_parquet(fs[0]).collect_schema()
    print("   columns:", list(sch.items()))

# sample rows of round_end and bomb_state (first file, lazy head)
re_head = pl.scan_parquet(files_of("round_end")[0]).head(5).collect()
print("== round_end head:")
print(re_head)
bs_head = pl.scan_parquet(files_of("bomb_state")[0]).head(5).collect()
print("== bomb_state head:")
print(bs_head)
rs_head = pl.scan_parquet(files_of("round_state")[0]).head(5).collect()
print("== round_state head:")
print(rs_head)
print("DONE", flush=True)
