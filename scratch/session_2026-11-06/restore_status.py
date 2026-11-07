"""Restore the original basic_player_status tomes onto EBS. No S3 writes.

For each chunk: download the raw per-match player_status files (GET only) to NVMe scratch, build the tome with the
pureskillgg_dsdk builder (same call as header_full.py), then check that the tome's row count equals the sum of the raw
files' row counts. Output: ~/adam/restore/chunks/cNNN/tome/csds/basic_player_status.<range>/ (range as in the S3 chunk name).
Manifest per chunk: ~/adam/restore/manifest/cNNN.json

  ~/venv-dsdk/bin/python restore_status.py --chunks c000
  ~/venv-dsdk/bin/python restore_status.py            (all 68 chunks)
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import polars as pl
import pyarrow.parquet as pq
from botocore.config import Config
from pureskillgg_dsdk.tome.builder import build_basic_tomes_from_fs

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
CH = "player_status"
OUT = os.path.expanduser("~/adam/restore")
SCR = "/mnt/sagemaker-nvme/restore"
BUILDER = os.path.expanduser("~/adam/builder")


def client():
    return boto3.client("s3", config=Config(max_pool_connections=128, retries={"max_attempts": 8},
                                            request_checksum_calculation="when_required",
                                            response_checksum_validation="when_required"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def s3_chunk_name(c, cli):
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/{c}/tome/csds/basic_{CH}.", Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            return p["Prefix"].rstrip("/").split("/")[-1]
    return None


# The dsdk builder needs the match keys (header and csds) as well as the channel itself, as in header_full.py.
FILES = [CH, "header", "csds"]


def fetch_one_file(cli, folder, name, root):
    dst = os.path.join(root, folder, name)
    if os.path.exists(dst):
        return "ok"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        cli.download_file(BUCKET, f"data/{folder}/{name}", dst + ".part")
    except Exception as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if os.path.exists(dst + ".part"):
            os.remove(dst + ".part")
        return "missing" if code in ("404", "NoSuchKey", "NotFound") else "failed"
    os.replace(dst + ".part", dst)
    return "ok"


def fetch(cli, mk, root):
    folder = mk.rsplit("/", 1)[0]
    status = {n: fetch_one_file(cli, folder, n, root) for n in FILES}
    if "failed" in status.values():
        return "failed"
    return status[CH]


def restore(c, cli):
    t0 = time.time()
    name = s3_chunk_name(c, cli)
    matches = pl.read_parquet(os.path.join(BUILDER, "data", c, "matches.parquet"), columns=["match_key"])["match_key"].to_list()
    build_root = os.path.join(SCR, "build", c)
    tome_root = os.path.join(OUT, "chunks", c)
    shutil.rmtree(build_root, ignore_errors=True)
    shutil.rmtree(tome_root, ignore_errors=True)
    os.makedirs(build_root)
    with ThreadPoolExecutor(64) as ex:
        res = list(ex.map(lambda m: fetch(cli, m, build_root), matches))
    counts = {k: res.count(k) for k in ("ok", "missing", "failed")}
    t_fetch = round(time.time() - t0)
    raw_files = glob.glob(os.path.join(build_root, "csds", "*", "*", "*", "*", CH))
    raw_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in raw_files)
    os.makedirs(tome_root, exist_ok=True)
    build_basic_tomes_from_fs([CH], ds_type="csds", tome_collection_root_path=tome_root, ds_collection_root_path=build_root)
    built = [d for d in glob.glob(os.path.join(tome_root, "tome", "csds", f"basic_{CH}*")) if os.path.isdir(d)]
    built_names = [os.path.basename(d) for d in built]
    if len(built) == 1 and name and built_names[0] != name:
        os.rename(built[0], os.path.join(os.path.dirname(built[0]), name))
        built_names = [name]
    tome_files = [f for d in built for f in glob.glob(os.path.join(d, "dataframe_*"))]
    tome_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in tome_files)
    manifest = {"chunk": c, "s3_name": name, "built_names": built_names, "matches": len(matches), "fetch": counts,
                "raw_files": len(raw_files), "raw_rows": raw_rows, "tome_files": len(tome_files), "tome_rows": tome_rows,
                "rows_equal": tome_rows == raw_rows, "fetch_secs": t_fetch, "total_secs": round(time.time() - t0)}
    os.makedirs(os.path.join(OUT, "manifest"), exist_ok=True)
    with open(os.path.join(OUT, "manifest", f"{c}.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    shutil.rmtree(build_root, ignore_errors=True)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="")
    a = ap.parse_args()
    cli = client()
    chunks = [c for c in a.chunks.split(",") if c] or [f"c{i:03d}" for i in range(68)]
    bad = []
    for c in chunks:
        m = restore(c, cli)
        log(c, json.dumps({k: m[k] for k in ("s3_name", "built_names", "matches", "fetch", "raw_rows", "tome_rows", "rows_equal", "total_secs")}))
        if not m["rows_equal"] or m["fetch"]["failed"]:
            bad.append(c)
    log("finished", len(chunks), "chunks; not equal or failed:", bad)


if __name__ == "__main__":
    main()
