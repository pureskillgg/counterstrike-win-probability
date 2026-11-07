import boto3, collections
from botocore.config import Config
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/fix332/"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 5, "mode": "standard"}))
keys = [l.strip() for l in open(W + "affected_keys.txt") if l.strip()]
res = collections.Counter()
absent = []
for mk in keys:
    folder = mk[: -len("/csds")] if mk.endswith("/csds") else mk
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"data/{folder}/player_status", MaxKeys=1)
    ok = r.get("KeyCount", 0) > 0
    r2 = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"data/{folder}/player_info", MaxKeys=1)
    ok2 = r2.get("KeyCount", 0) > 0
    res[(ok, ok2)] += 1
    if not (ok and ok2):
        absent.append(mk)
print("affected matches", len(keys), "| (player_status present, player_info present) counts:", dict(res))
print("absent examples", absent[:5])
