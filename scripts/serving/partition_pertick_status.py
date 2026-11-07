"""Split the per-tick player status of the October 1-8 test matches into one folder per match, in one pass over the tome's pages.

The parity test reads one match at a time; scanning every page for every match took too long. Output:
  E:\pureskill_tomes\pertick_status_1001_1008\per_match\<match_id>\part_<page>.parquet

  ...python.exe -I -X utf8 scripts\\serving\\partition_pertick_status.py
"""
import glob
import hashlib
import os
import sys
import time

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
PERTICK = r"E:\pureskill_tomes\pertick_status_1001_1008\tome\csds"
OUT = r"E:\pureskill_tomes\pertick_status_1001_1008\per_match"


def match_dir(key):
    return os.path.join(OUT, hashlib.sha1(key.encode("utf-8")).hexdigest()[:16])


def main():
    t0 = time.time()
    hdr = pl.concat([pl.read_parquet(f, columns=["key"]) for f in sorted(glob.glob(os.path.join(TOME, "header", "dataframe_*")))])
    grid = pl.read_parquet(os.path.join(ROOT, "data", "combined", "export_winprob_v8_ensemble_test_predictions.parquet"), columns=["match_key"])
    keys = sorted(set(hdr["key"].to_list()) & set(grid["match_key"].unique().to_list()))
    print(f"test matches in the tome: {len(keys)}", flush=True)
    keyset = pl.Series(keys)
    pages = sorted(glob.glob(os.path.join(PERTICK, "basic_player_status", "dataframe_*")))
    print(f"status pages: {len(pages)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    written = 0
    for i, page in enumerate(pages):
        df = pl.read_parquet(page).filter(pl.col("match_key").is_in(keyset))
        if df.height == 0:
            continue
        for (key,), part in df.partition_by("match_key", as_dict=True).items():
            d = match_dir(key)
            os.makedirs(d, exist_ok=True)
            part.write_parquet(os.path.join(d, f"part_{i:05d}.parquet"))
            written += part.height
        if (i + 1) % 300 == 0:
            print(f"  page {i + 1}/{len(pages)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {written:,} player-tick rows for {len(keys)} matches ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
