import boto3, json, random, hashlib, csv
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config

B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
cfg = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=32)
s3 = boto3.client("s3", config=cfg)

pairs = []
with open(W + "skeptic_s3_folders.tsv") as f:
    next(f)
    for line in f:
        y, m, d, u = line.rstrip("\n").split("\t")
        if y == "2026" and m == "10" and d in ("21", "22", "23", "24", "25"):
            pairs.append((d, u))
random.seed(7)
sample = random.sample(pairs, 60)


def one(item):
    d, u = item
    pfx = f"data/csds/2026/10/{d}/{u}/"
    files = {}
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=B, Prefix=pfx):
        for o in page.get("Contents", []):
            files[o["Key"][len(pfx):]] = o["Size"]
    pi = None
    if "player_info" in files:
        body = s3.get_object(Bucket=B, Key=pfx + "player_info")["Body"].read()
        pi = hashlib.md5(body).hexdigest()
    return {"day": d, "uuid": u, "nfiles": len(files), "files": files, "player_info_md5": pi}


with ThreadPoolExecutor(max_workers=16) as ex:
    res = list(ex.map(one, sample))
with open(W + "skeptic_oct21_sample.json", "w") as f:
    json.dump(res, f)
print("sampled", len(res), "files_per_folder_min_max", min(r["nfiles"] for r in res), max(r["nfiles"] for r in res), flush=True)
print("SAMPLE_DONE", flush=True)
