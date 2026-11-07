"""Coverage of the reduced player_status chunk set against the laptop grid, whole archive window (Oct 9 to Oct 25).

For every grid match in the window: is it in the admin key list (so it has a chunk)? If not, does its raw player_status
exist on S3? Read-only on S3 (head_object only). Writes coverage_missing.csv and coverage_summary.json next to this script.
"""
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
ADMIN = os.path.expanduser("~/tomes/tome/csds/header")
COMB = os.path.expanduser("~/adam/status/data/combined")
HERE = os.path.expanduser("~/adam/status_scripts/parity")
LO, HI = "2026/10/09", "2026/10/25"
s3 = boto3.client("s3", config=Config(max_pool_connections=64, retries={"max_attempts": 5}))


def admin_keys():
    fs = sorted(glob.glob(os.path.join(ADMIN, "dataframe_*")))
    return sorted(set(pl.concat([pl.read_parquet(f, columns=["key"]) for f in fs])["key"].to_list()))


def day_of(mk):
    p = mk.split("/")  # csds/yyyy/mm/dd/<id>/csds
    return f"{p[1]}/{p[2]}/{p[3]}"


def raw_exists(mk):
    folder = mk.rsplit("/", 1)[0]
    try:
        s3.head_object(Bucket=BUCKET, Key=f"data/{folder}/player_status")
        return True
    except ClientError:
        return False


def main():
    admin = admin_keys()
    admin_win = [k for k in admin if LO <= day_of(k) <= HI]
    G = (pl.read_parquet(os.path.join(COMB, "grid.parquet"), columns=["match_key", "era", "upload_day"])
         .filter((pl.col("era") == "2026") & (pl.col("upload_day") >= "2026/10/09") & (pl.col("upload_day") <= HI))
         .unique(subset=["match_key"]))
    grid = set(G["match_key"].to_list())
    admin_set = set(admin)
    missing = G.filter(~pl.col("match_key").is_in(list(admin_set)))
    admin_not_grid = [k for k in admin_win if k not in grid]
    print(f"grid matches in window {G.height}; admin keys in window {len(admin_win)}", flush=True)
    print(f"grid matches with no chunk entry: {missing.height}", flush=True)
    print("missing by day:", missing.group_by("upload_day").agg(pl.len().alias("n")).sort("upload_day").rows(),
          flush=True)
    with ThreadPoolExecutor(48) as ex:
        exists = list(ex.map(raw_exists, missing["match_key"].to_list()))
    missing = missing.with_columns(pl.Series("raw_player_status_on_s3", exists))
    n_raw = int(sum(exists))
    print(f"no-chunk matches with raw player_status on S3: {n_raw}; without: {missing.height - n_raw}", flush=True)
    print(f"admin keys in window that the grid does not have: {len(admin_not_grid)}", flush=True)
    missing.write_csv(os.path.join(HERE, "coverage_missing.csv"))
    summary = {"grid_matches_in_window": G.height, "admin_keys_in_window": len(admin_win),
               "grid_no_chunk": missing.height, "grid_no_chunk_raw_on_s3": n_raw,
               "grid_no_chunk_without_raw": missing.height - n_raw,
               "admin_in_window_not_in_grid": len(admin_not_grid),
               "missing_by_day": missing.group_by("upload_day").agg(pl.len().alias("n")).sort("upload_day").rows()}
    with open(os.path.join(HERE, "coverage_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("wrote coverage_summary.json", flush=True)


if __name__ == "__main__":
    main()
