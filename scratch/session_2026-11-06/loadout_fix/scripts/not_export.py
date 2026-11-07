import io, json, collections
import boto3
import pyarrow.parquet as pq
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
F = "/home/sagemaker-user/adam/builder/final"
s3 = boto3.client("s3", config=Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "standard"}))
EX = ThreadPoolExecutor(16)


def body(key):
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def keysets_all():
    out = []
    for i in range(68):
        base = f"adam/tomes/chunks/c{i:03d}/tome/csds/basic_player_status."
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=base):
            for o in page.get("Contents", []):
                if o["Key"].rsplit("/", 1)[1].startswith("keyset_"):
                    out.append((i, o["Key"]))
    return out


def keys_of(item):
    i, key = item
    t = pq.read_table(io.BytesIO(body(key)))
    col = "_" if "_" in t.column_names else t.column_names[0]
    return i, t.column(col).to_pylist()


chan = {}
for i, vals in EX.map(keys_of, keysets_all()):
    for k in vals:
        chan[k] = i
ek = set(json.load(open(F + "/export_clean_v2/match_keys.json")))
notex = [k for k in chan if k not in ek]
print("channel keys not in export:", len(notex), "| by chunk range:",
      "c000-c046", sum(1 for k in notex if chan[k] <= 46), "c047-c067", sum(1 for k in notex if chan[k] >= 47))

bm = pq.read_table(F + "/bucket_matches.parquet", columns=["match_key", "in_scope", "dup_keeper", "dup_group_size", "scope_reason", "map_name"]).to_pylist()
bmd = {r["match_key"]: r for r in bm}
c = collections.Counter()
for k in notex:
    r = bmd.get(k)
    c[(r["in_scope"], r["dup_keeper"], r["dup_group_size"] if r["dup_group_size"] is None else min(int(r["dup_group_size"]), 2))] += 1
print("not-in-export keys by (in_scope, dup_keeper, dup_group_size capped at 2):")
for kk, v in sorted(c.items(), key=lambda x: -x[1]):
    print("  ", kk, v)
sr = collections.Counter(str(bmd[k]["scope_reason"])[:60] for k in notex)
print("scope_reason top:", sr.most_common(6))
mp = collections.Counter(str(bmd[k]["map_name"]) for k in notex)
print("maps:", mp.most_common(9))
