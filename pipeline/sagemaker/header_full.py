import glob, json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
import boto3
import polars as pl
from botocore.config import Config
from pureskillgg_dsdk.tome.builder import build_basic_tomes_from_fs

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
BUILD = os.path.expanduser("~/header_build")
TOME = os.path.expanduser("~/header_tome")
ADMIN = os.path.expanduser("~/tomes/tome/csds/header")
FILES = ["csds", "header", "round_end"]
UPLOAD_PREFIX = "adam/tomes/header_build"


def admin_frame():
    fs = sorted(glob.glob(os.path.join(ADMIN, "dataframe_*")))
    return pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")


def fetch(key):
    folder = key.rsplit("/", 1)[0]
    s3 = FETCHER
    errs = []
    for f in FILES:
        dst = os.path.join(BUILD, folder, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            s3.download_file(BUCKET, f"data/{folder}/{f}", dst)
        except Exception as e:
            errs.append(f"{f}: {repr(e)[:160]}")
    return key, errs


FETCHER = None


def main():
    global FETCHER
    t0 = time.time()
    admin = admin_frame()
    keys = sorted(set(admin["key"].to_list()))
    print(f"ADMIN keys {len(keys)}", flush=True)
    shutil.rmtree(BUILD, ignore_errors=True)
    FETCHER = boto3.client("s3", config=Config(max_pool_connections=128, retries={"max_attempts": 5}))
    failures = []
    done = 0
    with ThreadPoolExecutor(64) as ex:
        for key, errs in ex.map(fetch, keys):
            done += 1
            if errs:
                failures.append((key, errs))
            if done % 5000 == 0:
                print(f"FETCH {done}/{len(keys)} secs={time.time() - t0:.0f} failures={len(failures)}", flush=True)
    print(f"FETCH DONE {done} failures={len(failures)} secs={time.time() - t0:.0f}", flush=True)
    if failures:
        print("FAILURES", json.dumps(failures[:10]), flush=True)
        sys.exit(2)

    shutil.rmtree(TOME, ignore_errors=True)
    t1 = time.time()
    build_basic_tomes_from_fs(["round_end"], ds_type="csds", tome_collection_root_path=TOME,
                              ds_collection_root_path=BUILD)
    print(f"BUILD DONE secs={time.time() - t1:.0f}", flush=True)

    hp = sorted(glob.glob(os.path.join(TOME, "tome", "csds", "header", "dataframe_*")))
    h = pl.concat([pl.read_parquet(f) for f in hp], how="diagonal_relaxed")
    built = set(h["key"].to_list())
    admin_set = set(keys)
    mr = h["max_rounds"].drop_nulls()
    joined = h.select(["key", "map_name", "match_date", "t_starters_score_final", "ct_starters_score_final"]).join(
        admin.select(["key", "map_name", "match_date", "t_starters_score_final", "ct_starters_score_final"]),
        on="key", how="inner", suffix="_admin")
    same = joined.filter((pl.col("map_name") == pl.col("map_name_admin")) &
                         (pl.col("match_date") == pl.col("match_date_admin")) &
                         (pl.col("t_starters_score_final") == pl.col("t_starters_score_final_admin")) &
                         (pl.col("ct_starters_score_final") == pl.col("ct_starters_score_final_admin"))).height
    summary = {"header_rows": h.height, "distinct_keys": len(built), "admin_keys": len(admin_set),
               "built_and_admin": len(built & admin_set), "built_only": len(built - admin_set),
               "admin_only": len(admin_set - built), "max_rounds_filled": mr.len(),
               "max_rounds_values": {str(k): v for k, v in zip(*[mr.value_counts()[c].to_list() for c in mr.value_counts().columns])} if mr.len() else {},
               "overlap_fields_equal": same, "overlap_rows": joined.height}
    print("VERIFY " + json.dumps(summary), flush=True)
    ok = (h.height == len(admin_set) == len(built) and not (built ^ admin_set) and summary["overlap_fields_equal"] == joined.height)
    if not ok:
        print("VERIFY FAILED: not uploading", flush=True)
        sys.exit(3)

    s3 = boto3.client("s3", config=Config(max_pool_connections=32))
    t2 = time.time()
    n = 0
    for root, _, files in os.walk(TOME):
        for name in files:
            local = os.path.join(root, name)
            rel = os.path.relpath(local, TOME)
            s3.upload_file(local, BUCKET, f"{UPLOAD_PREFIX}/{rel}")
            n += 1
    print(f"UPLOAD files={n} secs={time.time() - t2:.0f} prefix=s3://{BUCKET}/{UPLOAD_PREFIX}/", flush=True)
    print(f"ALL DONE total_secs={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
