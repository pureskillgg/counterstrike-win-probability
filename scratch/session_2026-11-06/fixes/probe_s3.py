import boto3, json
from botocore.config import Config
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}))
out = {}
for c in ["c000", "c061", "c062", "c063", "c064", "c065", "c066", "c067", "c068"]:
    p = f"{PREFIX}/{c}/tome/csds/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=p, Delimiter="/")
    subs = [x["Prefix"] for x in resp.get("CommonPrefixes", [])]
    info = {"subfolders": subs}
    for kind in ["basic_player_info", "basic_player_status"]:
        keys = []
        size = 0
        folders = set()
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/{c}/tome/csds/{kind}."):
            for o in page.get("Contents", []):
                keys.append(o["Key"]); size += o["Size"]
                folders.add(o["Key"].rsplit("/", 1)[0])
        info[kind] = {"objects": len(keys), "bytes": size, "folders": sorted(folders)}
    out[c] = info
print(json.dumps(out, indent=1))
