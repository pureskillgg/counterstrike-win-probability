"""Faster version of add_player_status_v3.py. Same rules, same outputs.

Per chunk (1,000 matches):
  - The next chunk is fetched in a background thread while this chunk is processed.
  - The reduce is one vectorised polars pass over all of the chunk's files, not one pass per match:
      one tick per second (freeze_end + 64*k inside the round), living players only (health > 0), KEEP columns only.
  - The reduced rows are split back into one file per match, because the team's builder reads per match.
  - The builder, the upload and STATUS_DONE.json are as in v3.

Usage (JupyterLab terminal, foreground):
    ~/venv-dsdk/bin/python add_player_status_v4.py --chunks 0-67
    ~/venv-dsdk/bin/python add_player_status_v4.py --chunks 50 --dry    (reduce only, prints the reduced row count)
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
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config
from botocore.exceptions import ClientError
from pureskillgg_dsdk.tome.builder import build_basic_tomes_from_fs

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
ADMIN = os.path.expanduser("~/tomes/tome/csds/header")
SCRATCH = "/mnt/sagemaker-nvme/status_add"
CHUNK = 1000
CHANNEL = "player_status"
TICK_STEP = 64
INV = ["inv_primary", "inv_secondary", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov",
       "inv_incgrenade", "inv_decoy", "inv_c4"]
KEEP = ["tick", "round", "player_id", "health", "armor", "has_helmet", "has_defuser", "money",
        "equipment_value_calc", "is_defusing"] + INV
FILES = [CHANNEL, "round_state", "round_end", "header", "csds"]
FETCH_THREADS = 96
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
        return o["Key"].split("/tome/csds/")[1].split("/")[0].split(".", 1)[1]
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


def fetch_chunk(i, root):
    t0 = time.time()
    matches = chunk_matches(i)
    counts = {"ok": 0, "missing": 0, "failed": 0}
    with ThreadPoolExecutor(FETCH_THREADS) as ex:
        for status, _ in ex.map(fetch_one, [(m, root) for m in matches]):
            counts[status] += 1
    return {"chunk": i, "matches": matches, "root": root, "counts": counts, "fetch_secs": round(time.time() - t0)}


def match_key_expr(col):
    # the file path ends in <yyyy>/<mm>/<dd>/<id>/<channel>; the match key is csds/<yyyy>/<mm>/<dd>/<id>/csds
    return pl.col(col).str.extract(r"(csds/\d{4}/\d{2}/\d{2}/[^/]+)/[^/]+$", 1) + "/csds"


def reduce_chunk(fetched):
    """One vectorised pass over the chunk's files, then one file per match written back for the builder."""
    root, matches = fetched["root"], fetched["matches"]
    folders = [m.rsplit("/", 1)[0] for m in matches]

    def paths(name):
        return [os.path.join(root, f, name) for f in folders if os.path.exists(os.path.join(root, f, name))]

    rs = (pl.scan_parquet(paths("round_state"), include_file_paths="src", extra_columns="ignore")
          .select(["src", "round", "tick", "event_type"])
          .filter(pl.col("event_type") == "round_freeze_end")
          .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), match_key_expr("src").alias("mk"))
          .group_by(["mk", "round"]).agg(pl.col("tick").min().alias("freeze")))
    re_ = (pl.scan_parquet(paths("round_end"), include_file_paths="src", extra_columns="ignore")
           .select(["src", "round", "tick"])
           .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), match_key_expr("src").alias("mk"))
           .group_by(["mk", "round"]).agg(pl.col("tick").min().alias("end")))
    rt = rs.join(re_, on=["mk", "round"], how="inner").collect().lazy()

    ps = (pl.scan_parquet(paths(CHANNEL), include_file_paths="src", extra_columns="ignore")
          .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), match_key_expr("src").alias("mk"))
          .join(rt, on=["mk", "round"], how="inner")
          .filter((pl.col("tick") >= pl.col("freeze")) & (pl.col("tick") < pl.col("end"))
                  & ((pl.col("tick") - pl.col("freeze")) % TICK_STEP == 0)
                  & (pl.col("health") > 0))
          .select(["mk"] + KEEP))
    out = ps.collect()
    reduced_rows = out.height

    schema = {c: out.schema[c] for c in KEEP}
    by_mk = {k[0]: v.select(KEEP) for k, v in out.partition_by("mk", as_dict=True).items()}
    empty = pl.DataFrame(schema=schema)
    for m in matches:
        folder = m.rsplit("/", 1)[0]
        dst = os.path.join(root, folder, CHANNEL)
        if not os.path.exists(os.path.dirname(dst)):
            continue  # the match had no files at all
        (by_mk.get(m, empty)).write_parquet(dst)
    return reduced_rows


def find_channel_dir(troot):
    for d, _, files in os.walk(os.path.join(troot, "tome", "csds")):
        if os.path.basename(d).startswith(f"basic_{CHANNEL}.") and any(f.startswith("dataframe_") for f in files):
            return d
    return None


def upload_one(args):
    local, key = args
    s3.upload_file(local, BUCKET, key)
    return key


def build_and_upload(i, fetched, suffix, reduced_rows, dry):
    root = fetched["root"]
    troot = os.path.join(SCRATCH, f"tome_c{i:03d}")
    shutil.rmtree(troot, ignore_errors=True)
    t1 = time.time()
    build_basic_tomes_from_fs([CHANNEL], ds_type="csds", tome_collection_root_path=troot,
                              ds_collection_root_path=root, read_threads=READ_THREADS)
    built = None
    for _ in range(60):
        built = find_channel_dir(troot)
        if built is not None:
            break
        time.sleep(2)
    if built is None:
        print(f"STATUS BUILD EMPTY c{i:03d} {json.dumps(fetched['counts'])}", flush=True)
        return False
    final_dir = os.path.join(troot, "tome", "csds", f"basic_{CHANNEL}.{suffix}")
    os.makedirs(final_dir, exist_ok=True)
    for f in os.listdir(built):
        src = os.path.join(built, f)
        if os.path.isfile(src) and (f.startswith("dataframe_") or f.startswith("keyset_") or f == "tome"):
            shutil.move(src, os.path.join(final_dir, f))
    if dry:
        print(f"DRY c{i:03d} reduced_rows {reduced_rows} built_in {round(time.time() - t1)}s", flush=True)
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(troot, ignore_errors=True)
        return True
    uploads = []
    for name in os.listdir(final_dir):
        local = os.path.join(final_dir, name)
        if os.path.isfile(local):
            uploads.append((local, f"adam/tomes/chunks/c{i:03d}/tome/csds/basic_{CHANNEL}.{suffix}/{name}"))
    with ThreadPoolExecutor(UPLOAD_THREADS) as ex:
        list(ex.map(upload_one, uploads))
    summary = {"chunk": i, "matches": len(fetched["matches"]), "fetched": fetched["counts"]["ok"],
               "missing": fetched["counts"]["missing"], "failed": fetched["counts"]["failed"],
               "reduced_rows": reduced_rows, "files_uploaded": len(uploads), "fetch_secs": fetched["fetch_secs"],
               "build_upload_secs": round(time.time() - t1)}
    s3.put_object(Bucket=BUCKET, Key=done_key(i), Body=json.dumps(summary).encode())
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(troot, ignore_errors=True)
    print("STATUS DONE " + json.dumps(summary), flush=True)
    return True


def prepare(i):
    root = os.path.join(SCRATCH, f"c{i:03d}")
    shutil.rmtree(root, ignore_errors=True)
    fetched = fetch_chunk(i, root)
    fetched["suffix"] = chunk_suffix(i)
    return fetched


def parse_chunks(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    todo = [i for i in parse_chunks(a.chunks) if a.dry or not is_done(i)]
    for i in parse_chunks(a.chunks):
        if i not in todo:
            print(f"SKIP c{i:03d} already has the reduced player_status", flush=True)
    if not todo:
        return
    pre = ThreadPoolExecutor(1)
    fut = pre.submit(prepare, todo[0])
    for k, i in enumerate(todo):
        fetched = fut.result()
        if k + 1 < len(todo):
            fut = pre.submit(prepare, todo[k + 1])
        t0 = time.time()
        reduced = reduce_chunk(fetched)
        reduce_secs = round(time.time() - t0)
        print(f"REDUCED c{i:03d} rows {reduced} reduce_secs {reduce_secs}", flush=True)
        build_and_upload(i, fetched, fetched["suffix"], reduced, a.dry)


if __name__ == "__main__":
    main()
