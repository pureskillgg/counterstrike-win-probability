import boto3, json
from collections import Counter
EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
s3 = boto3.client("s3")
ex = json.load(open(EXP + "TRAINING_EXCLUDE.json"))
batch = set(ex["match_keys"])
hb = set(l.strip() for l in open("/home/sagemaker-user/adam/builder/final/holdout_b_keys.txt") if l.strip())
folders = {}
for d in ["21", "22", "23", "24", "25"]:
    prefix = "data/csds/2026/10/%s/" % d
    names = []
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            names.append(cp["Prefix"])
    folders[d] = names
    print("day", d, "folders", len(names), flush=True)
s3keys = set()
for d, names in folders.items():
    for p in names:
        uuid = p.rstrip("/").split("/")[-1]
        s3keys.add("csds/2026/10/%s/%s/csds" % (d, uuid))
print("S3 match keys 10/21-25", len(s3keys), flush=True)
print("S3 in batch", len(s3keys & batch), "S3 in HB", len(s3keys & hb), "batch missing from S3", len(batch - s3keys), flush=True)
extra = sorted(s3keys - batch - hb)
print("S3 extra (not batch, not HB)", len(extra), flush=True)
print("extra by day", Counter(k.split("/")[3] + "/" + k.split("/")[4] for k in extra), flush=True)
json.dump({"extra": extra, "s3_total": len(s3keys)}, open("/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/refute_s3_extra.json", "w"))
# object count sample for extra folders
import random
random.seed(1)
samp = random.sample(extra, min(5, len(extra)))
for k in samp:
    d = k.split("/")[3]; uuid = k.split("/")[4]
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="data/csds/2026/10/%s/%s/" % (d, uuid)):
        n += page.get("KeyCount", 0)
    print("extra sample objects", n, flush=True)
