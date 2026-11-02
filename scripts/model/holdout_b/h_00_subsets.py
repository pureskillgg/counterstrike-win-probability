"""S2 step 0: holdout key list and holdout-only subsets of the combined store (read-only inputs, new files under data/holdout_b/ only).

Writes data/holdout_b/holdout_keys.txt (one key per line, for extract_status_1hz.py --match-list) and holdout-only copies of
grid.parquet (era 2026), matches.parquet and rounds.parquet, used by make_snapshot_cache.py --store holdout_b.
Nothing is read for labels or scored. Label columns travel with the copied grid rows because the builder schema carries them.
"""
import hashlib
import json
import os
import sys

import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
OUT = os.path.join(ROOT, "data", "holdout_b")
KEYS = os.path.join(ROOT, "data", "combined", "holdout_22_25_test_keys.json")
os.makedirs(OUT, exist_ok=True)
raw = open(KEYS, "rb").read()
doc = json.loads(raw.decode("utf-8"))
H = doc["match_keys"]
print("sha256 raw bytes:", hashlib.sha256(raw).hexdigest())
print("sha256 LF-normalised text:", hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest())
print("n_matches", doc["n_matches"], "list length", len(H), "unique", len(set(H)))
assert len(H) == len(set(H)) == 653
with open(os.path.join(OUT, "holdout_keys.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(H) + "\n")
Hl = pl.Series("match_key", H)
src = os.path.join(ROOT, "data", "combined")
G = pl.scan_parquet(os.path.join(src, "grid.parquet")).filter(pl.col("match_key").is_in(Hl) & (pl.col("era") == "2026"))
G.sink_parquet(os.path.join(OUT, "grid.parquet"), compression="zstd")
M = pl.scan_parquet(os.path.join(src, "matches.parquet")).filter(pl.col("match_key").is_in(Hl))
M.sink_parquet(os.path.join(OUT, "matches.parquet"), compression="zstd")
R = pl.scan_parquet(os.path.join(src, "rounds.parquet")).filter(pl.col("match_key").is_in(Hl))
R.sink_parquet(os.path.join(OUT, "rounds.parquet"), compression="zstd")
for n in ["grid", "matches", "rounds"]:
    lf = pl.scan_parquet(os.path.join(OUT, n + ".parquet"))
    print(n, "rows", lf.select(pl.len()).collect().item(), "matches", lf.select(pl.col("match_key").n_unique()).collect().item())
print("grid era values in holdout copy:", pl.scan_parquet(os.path.join(OUT, "grid.parquet")).select(pl.col("era").unique()).collect()["era"].to_list())
print("done")
