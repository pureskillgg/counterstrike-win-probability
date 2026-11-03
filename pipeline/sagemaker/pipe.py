import os, sys, json, time, shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
import boto3
from botocore.config import Config

SRC = "pureskillgg-csgo-production-dataexchange-csds-final-0"
WORK = "pureskillgg-csgo-production-sagemaker-0"
DAY = "2026/10/14"
STAGE = "/mnt/sagemaker-nvme/pipe"
OUT_PREFIX = "adam/pipeline/test"
T, CT = 2, 3
WORKERS = 16


def list_ids(n):
    s3 = boto3.client("s3")
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=SRC, Prefix=f"csds/{DAY}/"):
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".zip")]
    keys.sort()
    return [k.rsplit("/", 1)[1][:-4] for k in keys[:n]]


def zip_path(root, mid):
    return os.path.join(root, "csds", DAY, f"{mid}.zip")


def download(root, ids):
    s3 = boto3.client("s3", config=Config(max_pool_connections=64))

    def get(mid):
        s3.download_file(SRC, f"csds/{DAY}/{mid}.zip", zip_path(root, mid))

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(get, ids))


def init_worker():
    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"]:
        os.environ[k] = "1"
    import pyarrow
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    from pureskillgg_dsdk import DsReaderFs, GameDsLoader  # noqa: F401


def process_one(args):
    root, mid = args
    from pureskillgg_dsdk import DsReaderFs, GameDsLoader
    key = f"csds/{DAY}/{mid}/csds"
    loader = GameDsLoader(reader=DsReaderFs(root_path=root, manifest_key=key))
    dfs = loader.get_channels([{"channel": "round_end"}, {"channel": "round_state"}])
    re_ = dfs["round_end"]
    return {"match_id": mid, "match_key": key, "rounds": int(len(re_)),
            "t_round_wins": int((re_["winner_team_code"] == T).sum()),
            "ct_round_wins": int((re_["winner_team_code"] == CT).sum()),
            "round_state_rows": int(len(dfs["round_state"]))}


def run_batch(pool, batch_no, ids):
    import pandas as pd
    from pureskillgg_dsdk.adx.unzip import extract_adx_zips
    batch_id = f"{DAY.replace('/', '-')}_b{batch_no:03d}"
    root = os.path.join(STAGE, batch_id)
    os.makedirs(os.path.join(root, "csds", DAY), exist_ok=True)

    t = time.time()
    download(root, ids)
    t_dl = time.time() - t

    t = time.time()
    extract_adx_zips(root, delete_zips=True)
    t_ex = time.time() - t

    t = time.time()
    recs = pool.map(process_one, [(root, m) for m in ids], chunksize=2)
    t_pr = time.time() - t

    out_local = os.path.join(root, f"{batch_id}.parquet")
    pd.DataFrame(recs).to_parquet(out_local, index=False)
    key = f"{OUT_PREFIX}/{DAY}/{batch_id}.parquet"

    s3 = boto3.client("s3")
    t = time.time()
    s3.upload_file(out_local, WORK, key)
    t_up = time.time() - t
    size_ok = s3.head_object(Bucket=WORK, Key=key)["ContentLength"] == os.path.getsize(out_local)
    rows_back = len(pd.read_parquet(f"s3://{WORK}/{key}"))
    verified = bool(size_ok and rows_back == len(ids))
    if not verified:
        raise RuntimeError(f"verification failed for {key}; staging kept at {root}")
    shutil.rmtree(root, ignore_errors=True)
    return {"batch": batch_id, "matches": len(ids), "download_s": round(t_dl, 1),
            "extract_s": round(t_ex, 1), "process_s": round(t_pr, 1), "upload_s": round(t_up, 1),
            "s3_key": f"s3://{WORK}/{key}", "rows_back": rows_back, "verified": verified}


if __name__ == "__main__":
    n_matches, batch = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(STAGE, exist_ok=True)
    ids = list_ids(n_matches)
    ctx = get_context("spawn")
    with ctx.Pool(WORKERS, initializer=init_worker) as pool:
        for b in range(0, len(ids), batch):
            try:
                res = run_batch(pool, b // batch, ids[b:b + batch])
            except Exception as e:
                res = {"batch": b // batch, "error": repr(e)}
            print("BATCH " + json.dumps(res), flush=True)
    print("DONE", flush=True)


