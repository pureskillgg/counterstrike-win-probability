import boto3, json, time
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config

B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
cfg = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=32)

pairs = []
with open(W + "skeptic_s3_folders.tsv") as f:
    next(f)
    for line in f:
        y, m, d, u = line.rstrip("\n").split("\t")
        if y == "2026" and m == "10" and d in ("21", "22", "23", "24", "25"):
            pairs.append((d, u))
print("folders", len(pairs), flush=True)


def one(item):
    s3 = boto3.client("s3", config=cfg)
    d, u = item
    pfx = f"data/csds/2026/10/{d}/{u}/"
    files = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=pfx):
        for o in page.get("Contents", []):
            files[o["Key"][len(pfx):]] = o["Size"]
    return {"day": d, "uuid": u, "files": files}


t0 = time.time()
with ThreadPoolExecutor(max_workers=24) as ex:
    res = list(ex.map(one, pairs))
with open(W + "skeptic_oct21_full.json", "w") as f:
    json.dump(res, f)
print("listed", len(res), "files total", sum(len(r["files"]) for r in res), round(time.time() - t0, 1), flush=True)
print("FULL_DONE", flush=True)
