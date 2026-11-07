import json, io, collections
import numpy as np
import polars as pl
import boto3
from botocore.config import Config
FIN = "/home/sagemaker-user/adam/builder/final/"
EXP = FIN + "export_clean_v4/"
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
keys = json.load(open(EXP + "match_keys.json"))
cm = pl.read_parquet(FIN + "clean_matches.parquet", columns=["match_key", "src", "in_scope", "month", "map_name", "channels_complete"])
lap = set(cm.filter(pl.col("src") == "laptop")["match_key"].to_list())
buck = set(cm.filter(pl.col("src") == "bucket")["match_key"].to_list())
exp = set(keys)
print("laptop match_keys in clean_matches", len(lap), "in export", len(lap & exp))
print("bucket match_keys in clean_matches", len(buck), "in export", len(buck & exp))
print("sample laptop keys", sorted(lap)[:3])
print("sample bucket keys", sorted(buck)[:3])
print("export keys not in clean_matches", len(exp - lap - buck))
print("month dist laptop", cm.filter(pl.col("src") == "laptop").group_by("month").len().sort("month").head(20).rows())
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 5, "mode": "standard"}))
pre = f"{PREFIX}/c000/tome/csds/"
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=pre, Delimiter="/")
print("c000 prefixes", [p["Prefix"] for p in resp.get("CommonPrefixes", [])][:20])
for kind in ["basic_player_info", "basic_player_status"]:
    p = f"{PREFIX}/c000/tome/csds/{kind}."
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=p, MaxKeys=5)
    print(kind, [o["Key"] for o in r.get("Contents", [])][:5], "count-ish", r.get("KeyCount"))
