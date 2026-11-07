import io, json, collections, sys
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


def list_keysets(i):
    base = f"adam/tomes/chunks/c{i:03d}/tome/csds/basic_player_status."
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=base):
        for o in page.get("Contents", []):
            n = o["Key"].rsplit("/", 1)[1]
            if n.startswith("keyset_"):
                out.append(o["Key"])
    return out


def keys_of(key):
    t = pq.read_table(io.BytesIO(body(key)))
    col = "_" if "_" in t.column_names else t.column_names[0]
    return t.column(col).to_pylist()


keysets = []
for i in range(68):
    keysets += list_keysets(i)
chan = set()
for vals in EX.map(keys_of, keysets):
    chan.update(vals)
print("keysets", len(keysets), "channel distinct keys", len(chan))

bm = set(pq.read_table(F + "/bucket_matches.parquet", columns=["match_key"]).column("match_key").to_pylist())
print("channel == bucket_matches:", chan == bm, "| channel-bucket", len(chan - bm), "| bucket-channel", len(bm - chan))

ek = json.load(open(F + "/export_clean_v2/match_keys.json"))
cm = pq.read_table(F + "/clean_matches.parquet", columns=["match_key", "src"]).to_pylist()
src = {r["match_key"]: r["src"] for r in cm}
print("export by src", collections.Counter(src[k] for k in ek))

lap = [k for k in ek if src[k] == "laptop"]
brk = collections.defaultdict(lambda: [0, 0])
for k in lap:
    m = k[4:11]
    brk[m][0 if k in chan else 1] += 1
print("laptop-sourced export keys by month: month: [in channel, not in channel]")
for m in sorted(brk):
    print("  ", m, brk[m])
print("laptop total", len(lap), "in channel", sum(v[0] for v in brk.values()), "not in channel", sum(v[1] for v in brk.values()))

bkt = [k for k in ek if src[k] == "bucket"]
print("bucket-sourced export keys in channel", sum(1 for k in bkt if k in chan), "of", len(bkt))
# export keys with no source row at all
print("export keys without src row", sum(1 for k in ek if k not in src))
