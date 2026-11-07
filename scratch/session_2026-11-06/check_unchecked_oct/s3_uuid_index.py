import boto3, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config
B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
cfg = Config(retries={"max_attempts": 8, "mode": "standard"}, max_pool_connections=32)
s3 = boto3.client("s3", config=cfg)

def pref(p):
    out = []; tok = None
    while True:
        kw = dict(Bucket=B, Prefix=p, Delimiter="/", MaxKeys=1000)
        if tok: kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out += [x["Prefix"] for x in r.get("CommonPrefixes", [])]
        if r.get("IsTruncated"): tok = r["NextContinuationToken"]
        else: break
    return out

t0 = time.time()
years = [p for p in pref("data/csds/")]
print("years", years, flush=True)
day_prefixes = []
for y in years:
    for m in pref(y):
        for d in pref(m):
            day_prefixes.append(d)
print("day prefixes", len(day_prefixes), round(time.time()-t0,1), "s", flush=True)

def uuids_of(d):
    return [(d, u) for u in pref(d)]

rows = []
with ThreadPoolExecutor(max_workers=24) as ex:
    for res in ex.map(uuids_of, day_prefixes):
        rows.extend(res)
print("uuid folders", len(rows), round(time.time()-t0,1), "s", flush=True)
with open(W + "s3_uuid_index.tsv", "w") as f:
    f.write("day_prefix\tuuid_prefix\n")
    for d, u in rows:
        f.write(f"{d}\t{u}\n")
print("wrote", W + "s3_uuid_index.tsv", flush=True)
