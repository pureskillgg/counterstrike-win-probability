"""Build the 2026 snapshot feature table once (grid rows + map one-hots + the 46 loadout features), so the robustness experiments
(exp_robustness.py) do not each recompute it.  Rows = exactly what export_model.py trains on (roster-defect rounds left out).

  ...cs2gpu python.exe -I -X utf8 scripts\\model\\make_snapshot_cache.py [--store combined]   ->  data/<store>/cache_snapshot_2026.parquet
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

import features_status as FS
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]
ap = argparse.ArgumentParser()
ap.add_argument("--store", default="combined")
ap.add_argument("--chunks", type=int, default=5, help="build the loadout features for this many interleaved groups of matches, one after the other")
ap.add_argument("--era", default="2026", help="grid era to build (2026 default; 2024 writes the staged February matches)")
ap.add_argument("--out-name", default="cache_snapshot_2026.parquet", help="output file in data/<store>/; the control reads the 2026 default, which must not change")
ap.add_argument("--status-dir", default=None, help="status folder for the loadout features (default data/status_1hz)")
args = ap.parse_args()
t0 = time.time()
# EVERYTHING is done per chunk of matches (grid rows, roster-defect and table-vs-events exclusions, map one-hots, loadout features) and the chunks are
# appended as row groups of ONE parquet file. Loading the 19 M-row grid first, as the one-shot version did, held 9.5 GB before any feature existed.
import pyarrow.parquet as pq

d = os.path.join(W.ROOT, "data", args.store)
DROP = ("label_source", "upload_day", "build_num", "score_lead_diag")
lf = pl.scan_parquet(os.path.join(d, "grid.parquet"))
cols = [c for c in lf.collect_schema().names() if c not in DROP]
lf = lf.select(cols).filter(pl.col("era") == args.era)
M = pl.read_parquet(os.path.join(d, "matches.parquet")).select("match_key", "match_date")
out = os.path.join(d, args.out_name)
if args.out_name == "cache_snapshot_2026.parquet" and args.era != "2026":
    raise SystemExit("refusing to write the control file name with a non-2026 era")
tmp = out + ".tmp"
keys = sorted(lf.select("match_key").unique().collect()["match_key"].to_list())
n_chunks = max(1, args.chunks)
writer, n_rows, n_cols, tot0, tot_roster, tot_incons = None, 0, 0, 0, 0, 0
for i in range(n_chunks):
    part = keys[i::n_chunks]  # interleaved, so every chunk spans all upload days
    Gk = lf.filter(pl.col("match_key").is_in(part)).collect().join(M, on="match_key", how="left")
    n0 = Gk.height
    Gk = W.drop_roster_defect_rounds(Gk, args.store)
    n1 = Gk.height
    Gk = FS.drop_inconsistent_rounds(Gk, args.status_dir)
    tot0, tot_roster, tot_incons = tot0 + n0, tot_roster + (n0 - n1), tot_incons + (n1 - Gk.height)
    Gk = Gk.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Int8).alias(f"map_is_{m}") for m in MAPS])
    Gk = Gk.with_columns(FS.loadout_features(Gk, args.status_dir))
    tbl = Gk.to_arrow()
    if writer is None:
        writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
    writer.write_table(tbl.cast(writer.schema))
    n_rows += Gk.height
    n_cols = Gk.width
    print(f"  chunk {i + 1}/{n_chunks}: {len(part):,} matches, {Gk.height:,} rows ({time.time() - t0:.0f}s)", flush=True)
    del Gk, tbl
writer.close()
os.replace(tmp, out)
print(f"2026 rows {tot0:,}; roster-defect rounds left out {tot_roster:,}; table-vs-events inconsistent rounds left out {tot_incons:,}")
print(f"{n_rows:,} rows x {n_cols} columns, {len(keys):,} matches -> {out} ({time.time() - t0:.0f}s)")
