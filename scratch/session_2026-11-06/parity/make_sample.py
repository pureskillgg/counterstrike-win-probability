"""Parity sample: two matches per upload day (Oct 9 to Oct 25, seeded), their rows from the laptop's status_1hz.

Reads only the laptop's status_1hz day files (read-only). Writes the sample to the scratchpad folder.
Days Oct 26 onward are the holdout window and are not read.
"""
import os
import random
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
SRC = r"G:\pureskill\csds-cs2\data\status_1hz"
OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20261103
PER_DAY = 2
DAYS = [f"2026-10-{d:02d}" for d in range(9, 26)]

rng = random.Random(SEED)
picks = []  # (upload_day, match_key)
for d in DAYS:
    fn = os.path.join(SRC, d + ".parquet")
    mks = sorted(pl.scan_parquet(fn).select("match_key").unique().collect()["match_key"].to_list())
    chosen = rng.sample(mks, min(PER_DAY, len(mks)))
    picks += [(d, m) for m in chosen]
    print(f"{d}: {len(mks)} matches, picked {chosen}", flush=True)

sel = [m for _, m in picks]
frames = []
for d in DAYS:
    fn = os.path.join(SRC, d + ".parquet")
    part = pl.scan_parquet(fn).filter(pl.col("match_key").is_in(sel)).collect()
    if part.height:
        frames.append(part.with_columns(pl.lit(d).alias("upload_day")))
out = pl.concat(frames)
out.write_parquet(os.path.join(OUT, "laptop_status_1hz_sample.parquet"))
pl.DataFrame({"upload_day": [d for d, _ in picks], "match_key": sel}).write_csv(os.path.join(OUT, "sample_keys.csv"))
print(f"sample rows {out.height}, matches {out.select(pl.col('match_key').n_unique()).item()}", flush=True)
print("columns", out.columns, flush=True)
