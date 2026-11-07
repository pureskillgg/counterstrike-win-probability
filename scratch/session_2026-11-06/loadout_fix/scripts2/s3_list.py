import boto3, collections, sys
from botocore.config import Config
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
s3 = boto3.client("s3", config=Config(max_pool_connections=16, retries={"max_attempts": 8, "mode": "standard"}))


def list_objs(prefix):
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            out.append((o["Key"], o["Size"]))
    return out


for i in range(68):
    c = "c%03d" % i
    objs = list_objs(f"{PREFIX}/{c}/tome/csds/")
    folders = collections.defaultdict(lambda: [0, 0, []])
    for k, s in objs:
        d, n = k.rsplit("/", 1)
        f = folders[d]
        f[0] += 1
        f[1] += s
        if n.startswith("dataframe_") or n.startswith("keyset_"):
            f[2].append(n)
    for d, (n, b, names) in sorted(folders.items()):
        sub = d.rsplit("/", 1)[1]
        print(c, sub, "files", n, "bytes", b, "pages", len(names))
    if not folders:
        print(c, "NO FOLDERS")
