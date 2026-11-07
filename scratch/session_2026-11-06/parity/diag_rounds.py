"""Why are 80 tome rounds absent from the laptop grid? Compare their rounds.parquet attributes with grid rounds.
Reads cached pages and the space's copies of grid.parquet and rounds.parquet. Writes nothing to S3.
"""
import glob
import os

import polars as pl

HERE = os.path.expanduser("~/adam/status_scripts/parity")
WORK = "/mnt/sagemaker-nvme/parity"
COMB = os.path.expanduser("~/adam/status/data/combined")
MK_R = ["match_key", "round"]
keys = pl.read_csv(os.path.join(HERE, "sample_keys.csv"))["match_key"].to_list()
import json
rep = json.load(open(os.path.join(HERE, "parity_report.json")))
in_admin = [m for m in keys if m not in set(rep["not_in_admin"])]

pages = sorted(glob.glob(os.path.join(WORK, "c*", "dataframe_*")))
V = (pl.concat([pl.scan_parquet(p).filter(pl.col("match_key").is_in(in_admin)).select(["match_key", "round"]).collect()
                for p in pages], how="vertical_relaxed")
     .with_columns(pl.col("round").cast(pl.Int64)).unique())
G = (pl.read_parquet(os.path.join(COMB, "grid.parquet"), columns=["match_key", "round"])
     .filter(pl.col("match_key").is_in(in_admin)).with_columns(pl.col("round").cast(pl.Int64)).unique())
extra = V.join(G, on=MK_R, how="anti")
print("extra rounds:", extra.height, "matches:", extra["match_key"].n_unique(), flush=True)
print("extra round numbers (sorted, counts):", extra["round"].value_counts().sort("round").rows(), flush=True)

R = pl.read_parquet(os.path.join(COMB, "rounds.parquet"))
print("rounds.parquet columns:", R.columns, flush=True)
R = R.with_columns(pl.col("round").cast(pl.Int64))
ex_attr = R.join(extra, on=MK_R, how="inner")
gr_attr = R.join(G, on=MK_R, how="inner")
print("extra rounds found in rounds.parquet:", ex_attr.height, "of", extra.height, flush=True)
print("extra-round rows (first 8):", ex_attr.head(8).to_dicts(), flush=True)
print("grid-round rows (first 3):", gr_attr.head(3).to_dicts(), flush=True)

# per-column: value counts among extra vs grid rounds (short categorical columns only)
for c in R.columns:
    if c in MK_R or R[c].dtype in (pl.Float32, pl.Float64):
        continue
    n_unique = R[c].n_unique()
    if n_unique > 12:
        continue
    e = ex_attr[c].value_counts().rows() if ex_attr.height else []
    g = gr_attr[c].value_counts().rows() if gr_attr.height else []
    print(f"col {c}: extra {sorted(e, key=str)} | grid {sorted(g, key=str)}", flush=True)
