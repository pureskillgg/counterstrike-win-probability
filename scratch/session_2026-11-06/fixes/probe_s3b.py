import boto3, json, re
from botocore.config import Config
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}))
out = {}
for i in range(0, 70):
    c = "c%03d" % i
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/{c}/tome/csds/", Delimiter="/", MaxKeys=1000)
    subs = [x["Prefix"] for x in resp.get("CommonPrefixes", [])]
    if not subs:
        out[c] = "NO_FOLDERS"
        continue
    st = [s for s in subs if "/basic_player_status." in s]
    inf = [s for s in subs if "/basic_player_info." in s]
    win = st[0].split("basic_player_status.")[1].rstrip("/") if st else None
    size_st = 0; n_st = 0
    if st:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=st[0]):
            for o in page.get("Contents", []):
                if "/header/" in o["Key"]: continue
                n_st += 1; size_st += o["Size"]
    out[c] = {"window": win, "n_status_pages": n_st, "status_MB": round(size_st/1e6,1), "has_info": bool(inf), "n_subfolders": len(subs)}
    if i in (0,):
        pass
for c, v in out.items():
    print(c, v)
