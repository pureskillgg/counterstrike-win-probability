"""Additive chunk update: add the raw player_status channel to each existing chunk tome, without rebuilding the other channels.

Reuses the team's chunk_build.py functions (admin keys, chunking, fetch_batch, build_basic_tomes_from_fs, verification and
upload). Only basic_player_status.<dates> is written into adam/tomes/chunks/cNNN/. Existing channel folders are not touched.
Resumable: a chunk with adam/tomes/chunks/cNNN/STATUS_DONE.json is skipped.

Usage (JupyterLab terminal, foreground):
    ~/venv-dsdk/bin/python add_player_status.py --chunks 50            (test on one chunk)
    ~/venv-dsdk/bin/python add_player_status.py --chunks 0-67          (all)
"""
import argparse
import glob
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import boto3
import polars as pl
from botocore.config import Config
from pureskillgg_dsdk.tome.builder import build_basic_tomes_from_fs

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
ADMIN = os.path.expanduser("~/tomes/tome/csds/header")
SCRATCH = "/mnt/sagemaker-nvme/status_add"
CHUNK = 1000
CHANNELS = ["player_status"]
FETCH_PROCS = 4
FETCH_THREADS = 64
READ_THREADS = 16
WANTED = {"player_status", "csds", "header"}
s3 = boto3.client("s3", config=Config(max_pool_connections=128, retries={"max_attempts": 5}))


def admin_keys():
    fs = sorted(glob.glob(os.path.join(ADMIN, "dataframe_*")))
    return sorted(set(pl.concat([pl.read_parquet(f, columns=["key"]) for f in fs])["key"].to_list()))


def chunk_keys(i):
    keys = admin_keys()
    return keys[i * CHUNK:(i + 1) * CHUNK]


def done_key(i):
    return f"adam/tomes/chunks/c{i:03d}/STATUS_DONE.json"


def is_done(i):
    try:
        s3.head_object(Bucket=BUCKET, Key=done_key(i))
        return True
    except Exception:
        return False


def fetch_batch(args):
    root, batch = args
    client = boto3.client("s3", config=Config(max_pool_connections=128, retries={"max_attempts": 5}))

    def fetch_match(key):
        folder = key.rsplit("/", 1)[0]
        n = 0
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"data/{folder}/"):
            for o in page.get("Contents", []):
                name = o["Key"].rsplit("/", 1)[1]
                if name in WANTED:
                    dst = os.path.join(root, folder, name)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    body = client.get_object(Bucket=BUCKET, Key=o["Key"])["Body"]
                    with open(dst, "wb") as f:
                        shutil.copyfileobj(body, f)
                    n += 1
        return n

    with ThreadPoolExecutor(FETCH_THREADS) as ex:
        return sum(ex.map(fetch_match, batch))


def upload_channel(troot, prefix):
    count = 0
    for root, _, files in os.walk(troot):
        for name in files:
            local = os.path.join(root, name)
            rel = os.path.relpath(local, troot)
            s3.upload_file(local, BUCKET, f"{prefix}/{rel}")
            count += 1
    return count


def process_chunk(i):
    t0 = time.time()
    keys = chunk_keys(i)
    root = os.path.join(SCRATCH, f"c{i:03d}")
    troot = os.path.join(SCRATCH, f"tome_c{i:03d}")
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(troot, ignore_errors=True)
    batches = [keys[k::FETCH_PROCS] for k in range(FETCH_PROCS)]
    pool = get_context("spawn").Pool(FETCH_PROCS)
    files = sum(pool.map(fetch_batch, [(root, b) for b in batches if b]))
    pool.close()
    pool.join()
    t1 = time.time()
    build_basic_tomes_from_fs(CHANNELS, ds_type="csds", tome_collection_root_path=troot,
                              ds_collection_root_path=root, read_threads=READ_THREADS)
    built_dir = glob.glob(os.path.join(troot, "tome", "csds", "basic_player_status.*"))
    hp = sorted(glob.glob(os.path.join(built_dir[0], "dataframe_*"))) if built_dir else []
    built = set(pl.concat([pl.read_parquet(f, columns=["match_key"]) for f in hp])["match_key"].unique().to_list()) if hp else set()
    expected = {k.rsplit("/", 1)[0] for k in keys}
    summary = {"chunk": i, "matches": len(keys), "files_fetched": files, "matches_with_status": len(built & expected),
               "missing": len(expected - built), "fetch_secs": round(t1 - t0), "build_secs": round(time.time() - t1)}
    if not built_dir:
        print("STATUS BUILD EMPTY " + json.dumps(summary), flush=True)
        return False
    n = upload_channel(os.path.join(troot, "tome", "csds", os.path.basename(built_dir[0])),
                       f"adam/tomes/chunks/c{i:03d}/tome/csds/{os.path.basename(built_dir[0])}")
    summary["uploaded_files"] = n
    s3.put_object(Bucket=BUCKET, Key=done_key(i), Body=json.dumps(summary).encode())
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(troot, ignore_errors=True)
    print("STATUS DONE " + json.dumps(summary), flush=True)
    return True


def parse_chunks(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    for i in parse_chunks(a.chunks):
        if is_done(i):
            print(f"SKIP c{i:03d} already has player_status", flush=True)
            continue
        process_chunk(i)


if __name__ == "__main__":
    main()
