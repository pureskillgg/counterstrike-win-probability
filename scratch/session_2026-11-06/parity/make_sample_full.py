"""Widen the parity sample: every match of two full upload days (Oct 12 and Oct 21), plus the 34-match seeded sample.

Read-only on the laptop's status_1hz day files. Writes laptop_status_1hz_sample2.parquet and sample2_keys.csv here.
Days Oct 26 onward (holdout window) are not read.
"""
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
SRC = r"G:\pureskill\csds-cs2\data\status_1hz"
OUT = os.path.dirname(os.path.abspath(__file__))
FULL_DAYS = ["2026-10-12", "2026-10-21"]

seed_keys = pl.read_csv(os.path.join(OUT, "sample_keys.csv"))
full_pairs = []
for d in FULL_DAYS:
    mks = sorted(pl.scan_parquet(os.path.join(SRC, d + ".parquet")).select("match_key").unique().collect()["match_key"].to_list())
    full_pairs += [(d, m) for m in mks]
    print(f"{d}: all {len(mks)} matches", flush=True)
full = pl.DataFrame({"upload_day": [d for d, _ in full_pairs], "match_key": [m for _, m in full_pairs]})
keys = pl.concat([seed_keys.select("upload_day", "match_key"), full]).unique(subset=["match_key"], keep="first",
                                                                              maintain_order=True)
sel = keys["match_key"].to_list()
frames = []
for d in sorted(set(keys["upload_day"].to_list())):
    part = pl.scan_parquet(os.path.join(SRC, d + ".parquet")).filter(pl.col("match_key").is_in(sel)).collect()
    if part.height:
        frames.append(part.with_columns(pl.lit(d).alias("upload_day")))
out = pl.concat(frames)
out.write_parquet(os.path.join(OUT, "laptop_status_1hz_sample2.parquet"))
keys.write_csv(os.path.join(OUT, "sample2_keys.csv"))
print(f"sample2: matches {keys.height}, rows {out.height}", flush=True)
