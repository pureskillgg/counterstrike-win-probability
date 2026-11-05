"""Additive chunk update, minimal version: add a reduced player_status channel to each existing chunk tome.

Per match, before the tome is built, the raw player_status is reduced to the bare minimum the sequence inputs need:
  - one tick per second: ticks freeze_end + 64*k that fall inside the round (freeze_end <= tick < round end)
  - living players only (health > 0)
  - only the needed columns (identifiers, health, armour, helmet, defuser, money, equipment value, defusing, inventory)
Then the team's build_basic_tomes_from_fs builds the channel from the reduced files on NVMe scratch, the built channel is
placed under adam/tomes/chunks/cNNN/tome/csds/basic_player_status.<same dates suffix>/, and STATUS_DONE.json is written.
Existing channel folders are not touched. Resumable: a chunk with STATUS_DONE.json is skipped.

Usage (JupyterLab terminal, foreground):
    ~/venv-dsdk/bin/python add_player_status_v3.py --chunks 50
    ~/venv-dsdk/bin/python add_player_status_v3.py --chunks 0-67
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
TICK_STEP = 64  # one tick per second at 64 ticks per second
INV = ["inv_primary", "inv_secondary", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov",
       "inv_incgrenade", "inv_decoy", "inv_c4"]
KEEP = ["tick", "round", "player_id", "health", "armor", "has_helmet", "has_defuser", "money",
        "equipment_value_calc", "is_defusing"] + INV
FILES = [CHANNEL, "round_state", "round_end", "header", "csds"]  # round_state and round_end give the freeze and end ticks
FETCH_THREADS = 96
REDUCE_THREADS = 8
UPLOAD_THREADS = 32
READ_THREADS = 8
s3 = boto3.client("s3", config=Config(max_pool_connections=160, retries={"max_attempts": 5},
                                      request_checksum_calculation="when_required",
                                      response_checksum_validation="when_required"))


def admin_keys():
    fs = sorted(glob.glob(os.path.join(ADMIN, "dataframe_*")))
    return sorted(set(pl.concat([pl.read_parquet(f, columns=["key"]) for f in fs])["key"].to_list()))


def chunk_matches(i):
    return admin_keys()[i * CHUNK:(i + 1) * CHUNK]


def done_key(i):
    return f"adam/tomes/chunks/c{i:03d}/STATUS_DONE.json"


def is_done(i):
    try:
        s3.head_object(Bucket=BUCKET, Key=done_key(i))
        return True
    except ClientError:
        return False


def chunk_suffix(i):
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"adam/tomes/chunks/c{i:03d}/tome/csds/basic_", MaxKeys=50)
    for o in resp.get("Contents", []):
        return o["Key"].split("/tome/csds/")[1].split("/")[0].split(".", 1)[1]  # all basic_ channels in a chunk share one dates suffix
    raise RuntimeError(f"no basic_ channel folder in chunk c{i:03d}")


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


def reduce_one(args):
    match_key, root = args
    base = os.path.join(root, match_key.rsplit("/", 1)[0])
    try:
        rs = pl.read_parquet(os.path.join(base, "round_state"), columns=["round", "tick", "event_type"])
        fz = (rs.filter(pl.col("event_type") == "round_freeze_end").group_by("round")
              .agg(pl.col("tick").min().alias("freeze")))
        re_ = pl.read_parquet(os.path.join(base, "round_end"), columns=["round", "tick"])
        en = re_.group_by("round").agg(pl.col("tick").min().alias("end"))
        rt = fz.join(en, on="round", how="inner")
        ps = pl.read_parquet(os.path.join(base, CHANNEL))
        ps = (ps.join(rt, on="round", how="inner")
              .filter((pl.col("tick") >= pl.col("freeze")) & (pl.col("tick") < pl.col("end"))
                      & ((pl.col("tick") - pl.col("freeze")) % TICK_STEP == 0)
                      & (pl.col("health") > 0))
              .select(KEEP))
        ps.write_parquet(os.path.join(base, CHANNEL))
        return ("ok", match_key, ps.height)
    except Exception as exc:
        return ("failed", match_key, repr(exc)[:200])


def find_channel_dir(troot):
    for d, _, files in os.walk(os.path.join(troot, "tome", "csds")):
        if os.path.basename(d).startswith(f"basic_{CHANNEL}.") and any(f.startswith("dataframe_") for f in files):
            return d
    return None


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

    reduced_rows, reduce_failed = 0, 0
    with ThreadPoolExecutor(REDUCE_THREADS) as ex:
        for res in ex.map(reduce_one, [(m, root) for m in matches]):
            if res[0] == "ok":
                reduced_rows += res[2]
            else:
                reduce_failed += 1
    t2 = time.time()

    build_basic_tomes_from_fs([CHANNEL], ds_type="csds", tome_collection_root_path=troot,
                              ds_collection_root_path=root, read_threads=READ_THREADS)
    built = None
    for _ in range(60):  # the builder can finish writing its pages a moment after it returns
        built = find_channel_dir(troot)
        if built is not None and not any(
                os.path.exists(os.path.join(built, f)) and os.path.getsize(os.path.join(built, f)) == 0
                for f in os.listdir(built) if f.startswith("dataframe_")):
            break
        time.sleep(2)
    if built is None:
        seen = [(d, len(files)) for d, _, files in os.walk(troot)]
        print(f"STATUS BUILD EMPTY c{i:03d} {json.dumps(counts)} tree={json.dumps(seen[:12])}", flush=True)
        return False

    final_dir = os.path.join(troot, "tome", "csds", f"basic_{CHANNEL}.{suffix}")
    os.makedirs(final_dir, exist_ok=True)
    for f in os.listdir(built):
        src = os.path.join(built, f)
        if os.path.isfile(src) and (f.startswith("dataframe_") or f.startswith("keyset_") or f == "tome"):
            shutil.move(src, os.path.join(final_dir, f))

    uploads = []
    for name in os.listdir(final_dir):
        local = os.path.join(final_dir, name)
        if os.path.isfile(local):  # skip the builder's nested header folder
            uploads.append((local, f"adam/tomes/chunks/c{i:03d}/tome/csds/basic_{CHANNEL}.{suffix}/{name}"))
    with ThreadPoolExecutor(UPLOAD_THREADS) as ex:
        list(ex.map(upload_one, uploads))

    summary = {"chunk": i, "matches": len(matches), "fetched": counts["ok"], "missing": counts["missing"],
               "failed": counts["failed"], "reduced_rows": reduced_rows, "reduce_failed": reduce_failed,
               "files_uploaded": len(uploads), "fetch_secs": round(t1 - t0), "reduce_secs": round(t2 - t1),
               "build_upload_secs": round(time.time() - t2)}
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
            print(f"SKIP c{i:03d} already has the reduced player_status", flush=True)
            continue
        process_chunk(i)


if __name__ == "__main__":
    main()
