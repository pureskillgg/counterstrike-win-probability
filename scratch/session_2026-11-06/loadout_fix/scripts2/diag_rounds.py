import io, json, sys, collections
import numpy as np
import polars as pl
import boto3
from botocore.config import Config
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}))
keys = json.load(open(EXP + "match_keys.json"))
off = np.load(EXP + "offsets.npy").astype(np.int64)
rm = np.load(EXP + "round_match.npy").astype(np.int64)
rn = np.load(EXP + "round_num.npy").astype(np.int64)
key_m = {k: i for i, k in enumerate(keys)}
# export round numbers per match
exp_rounds = collections.defaultdict(set)
for m, r in zip(rm.tolist(), rn.tolist()):
    exp_rounds[m].add(r)
pages = []
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="adam/tomes/chunks/c037/tome/csds/basic_player_status."):
    for o in page.get("Contents", []):
        d, n = o["Key"].rsplit("/", 1)
        if d.rsplit("/", 1)[-1].startswith("basic_player_status.") and n.startswith("dataframe_"):
            pages.append(o["Key"])
pages = sorted(pages)[:1]
df = pl.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=pages[0])["Body"].read()), columns=["match_key", "round", "tick"])
print("page", pages[0], "rows", df.height, "dtype round", df.schema["round"])
g = df.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
cnt_in_export = 0
cnt_not = collections.Counter()
examples = []
matches_with_extra = 0
matches_total = 0
for mk, sub in g.group_by("match_key"):
    mk = mk[0]
    if mk not in key_m:
        cnt_not["match_not_in_export"] += 1
        continue
    m = key_m[mk]
    matches_total += 1
    ch = set(sub["round"].to_list())
    ex = exp_rounds[m]
    extra = sorted(ch - ex)
    missing = sorted(ex - ch)
    if extra:
        matches_with_extra += 1
        if len(examples) < 12:
            examples.append({"match_key": mk, "channel_rounds": sorted(ch)[:40], "export_rounds": sorted(ex)[:40], "extra": extra[:20], "missing": missing[:20]})
        for r in extra:
            cnt_not["extra_round_" + ("gt24" if r > 24 else "le24")] += 1
    if missing:
        cnt_not["matches_with_export_rounds_absent_in_channel"] += 1
print("matches in page", matches_total, "with extra channel rounds", matches_with_extra)
print(dict(cnt_not))
for e in examples:
    print(json.dumps(e))
