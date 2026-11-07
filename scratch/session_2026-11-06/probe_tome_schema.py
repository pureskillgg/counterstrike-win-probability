import io, json, time
import polars as pl
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
FIN = "/home/sagemaker-user/adam/builder/final/"
EXP = FIN + "export_clean_v4/"
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
s3 = boto3.client("s3", config=Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "standard"}))
POOL = ThreadPoolExecutor(12)
keys = set(json.load(open(EXP + "match_keys.json")))
cm = pl.read_parquet(FIN + "clean_matches.parquet", columns=["match_key", "src"])
lap = set(cm.filter(pl.col("src") == "laptop")["match_key"].to_list())
buck = set(cm.filter(pl.col("src") == "bucket")["match_key"].to_list())


def list_pages(c, kind):
    prefix = f"{PREFIX}/{c}/tome/csds/{kind}."
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            d, n = o["Key"].rsplit("/", 1)
            if d.rsplit("/", 1)[-1].startswith(kind + ".") and n.startswith("dataframe_"):
                out.append(o["Key"])
    return sorted(out)


def get_bytes(k):
    return s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()


# schema from one page each
ik = list_pages("c050", "basic_player_info")
sk = list_pages("c050", "basic_player_status")
print("c050 info pages", len(ik), "status pages", len(sk))
fi = pl.read_parquet(io.BytesIO(get_bytes(ik[0])))
print("INFO cols", fi.columns, fi.height)
print(fi.head(3))
fs = pl.read_parquet(io.BytesIO(get_bytes(sk[0])))
print("STATUS cols", fs.columns, fs.height)
print(fs.head(3))
print("status dtypes", dict(zip(fs.columns, [str(t) for t in fs.dtypes])))
# coverage of laptop vs bucket keys in 2026-09/10 chunks
t0 = time.time()
found = set()
for c in ["c%03d" % i for i in range(44, 63)]:
    ik2 = list_pages(c, "basic_player_info")
    if not ik2:
        continue
    for b in POOL.map(get_bytes, ik2):
        f = pl.read_parquet(io.BytesIO(b), columns=["match_key"])
        found.update(f["match_key"].unique().to_list())
print("chunks c044..c062 distinct info match_keys", len(found), "sec", round(time.time() - t0, 1))
print("found & laptop", len(found & lap), "of laptop", len(lap), "| found & bucket", len(found & buck), "| found & export", len(found & keys))
print("laptop keys sample", sorted(lap)[:2])
