"""Additive chunk update, efficient version: add the raw player_status channel to each existing chunk tome.

Per chunk (1,000 matches):
  1. Direct GET of data/<match folder>/player_status for every match (no listing), in one process with many threads.
     Missing keys are recorded, not retried. Other errors retry three times.
  2. Files stay on NVMe scratch.
  3. The team's build_basic_tomes_from_fs builds the player_status channel from the scratch files.
  4. The built channel folder is renamed to basic_player_status.<same dates suffix as the chunk's other channels>.
  5. Verify the matches in the built channel against the chunk's expected matches.
  6. Upload the new channel folder to adam/tomes/chunks/cNNN/tome/csds/. Existing channel folders are not touched.
  7. Write adam/tomes/chunks/cNNN/STATUS_DONE.json. A chunk with that marker is skipped (resumable).

Usage (JupyterLab terminal, foreground):
    ~/venv-dsdk/bin/python add_player_status_v2.py --chunks 50
    ~/venv-dsdk/bin/python add_player_status_v2.py --chunks 0-67
"""
import argparse
import glob
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError
from pureskillgg_dsdk.tome.builder import build_basic_tomes_from_fs

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
ADMIN = os.path.expanduser("~/tomes/tome/csds/header")
SCRATCH = "/mnt/sagemaker-nvme/status_add"
CHUNK = 1000
CHANNEL = "player_status"
FETCH_THREADS = 96
UPLOAD_THREADS = 32
READ_THREADS = 8
s3 = boto3.client("s3", config=Config(max_pool_connections=160, retries={"max_attempts": 5}))


def admin_keys():
    fs = sorted(glob.glob(os.path.join(ADMIN, "dataframe_*")))
    return sorted(set(pl.concat([pl.read_parquet(f, columns=["key"]) for f in fs])["key"].to_list()))


def chunk_matches(i):
    keys = admin_keys()
    return keys[i * CHUNK:(i + 1) * CHUNK]


def done_key(i):
    return f"adam/tomes/chunks/c{i:03d}/STATUS_DONE.json"


def is_done(i):
    try:
        s3.head_object(Bucket=BUCKET, Key=done_key(i))
        return True
    except ClientError:
        return False


def chunk_suffix(i):
    """The dates suffix the chunk's other channels use, e.g. '2026-10-10,2026-10-11', read from the chunk's S3 folder names."""
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"adam/tomes/chunks/c{i:03d}/tome/csds/basic_player_inputs.", MaxKeys=5)
    for o in resp.get("Contents", []):
        name = o["Key"].split("/tome/csds/")[1].split("/")[0]
        return name.split(".", 1)[1]
    raise RuntimeError(f"no basic_player_inputs folder found in chunk c{i:03d}")


FILES = [CHANNEL, "header", "csds"]  # the builder needs header and csds for each match, as in the team's chunk_build


def fetch_one(args):
    match_key, root = args
    folder = match_key.rsplit("/", 1)[0]
    status = "ok"
    for name in FILES:
        dst = os.path.join(root, folder, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(3):
            try:
                body = s3.get_object(Bucket=BUCKET, Key=f"data/{folder}/{name}")["Body"]
                with open(dst, "wb") as f:
                    shutil.copyfileobj(body, f)
                break
            except ClientError as e:
                if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                    status = "missing"
                    break
                time.sleep(2 * (attempt + 1))
        else:
            return ("failed", match_key)
    return (status, match_key)


def upload_one(args):
    local, key = args
    s3.upload_file(local, BUCKET, key)
    return key


def process_chunk(i):
    t0 = time.time()
    matches = chunk_matches(i)
    suffix = chunk_suffix(i)
    root = os.path.join(SCRATCH, f"c{i:03d}")
    troot = os.path.join(SCRATCH, f"tome_c{i:03d}")
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(troot, ignore_errors=True)

    counts = {"ok": 0, "missing": 0, "failed": 0}
    with ThreadPoolExecutor(FETCH_THREADS) as ex:
        for status, _ in ex.map(fetch_one, [(m, root) for m in matches]):
            counts[status] += 1
    t1 = time.time()

    build_basic_tomes_from_fs([CHANNEL], ds_type="csds", tome_collection_root_path=troot,
                              ds_collection_root_path=root, read_threads=READ_THREADS)
    hits = [d for d in glob.glob(os.path.join(troot, "tome", "csds", "**", CHANNEL), recursive=True) if os.path.isdir(d)]
    built_dir = hits[0] if hits else None
    if built_dir is None:
        print(f"STATUS BUILD EMPTY c{i:03d} {json.dumps(counts)}", flush=True)
        return False
    pages = sorted(glob.glob(os.path.join(built_dir, "dataframe_*")))
    built = set(pl.concat([pl.read_parquet(f, columns=["match_key"]) for f in pages])["match_key"].unique().to_list()) if pages else set()
    expected = set(matches)
    final_name = f"basic_{CHANNEL}.{suffix}"
    final_dir = os.path.join(troot, "tome", "csds", final_name)
    if os.path.abspath(built_dir) != os.path.abspath(final_dir):
        shutil.move(built_dir, final_dir)

    uploads = []
    for root_dir, _, files in os.walk(final_dir):
        for name in files:
            local = os.path.join(root_dir, name)
            rel = os.path.relpath(local, os.path.join(troot, "tome", "csds"))
            uploads.append((local, f"adam/tomes/chunks/c{i:03d}/tome/csds/{rel}"))
    with ThreadPoolExecutor(UPLOAD_THREADS) as ex:
        list(ex.map(upload_one, uploads))

    summary = {"chunk": i, "matches": len(matches), "fetched": counts["ok"], "missing": counts["missing"],
               "failed": counts["failed"], "built_matches": len(built & expected), "not_built": len(expected - built),
               "files_uploaded": len(uploads), "fetch_secs": round(t1 - t0), "build_and_upload_secs": round(time.time() - t1)}
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
