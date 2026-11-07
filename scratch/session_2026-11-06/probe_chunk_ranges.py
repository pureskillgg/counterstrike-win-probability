import collections, json
import boto3
from botocore.config import Config
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 5, "mode": "standard"}))
ranges = {}
for i in range(0, 63):
    c = "c%03d" % i
    p = f"{PREFIX}/{c}/tome/csds/basic_player_info."
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=p, Delimiter="/")
    names = [x["Prefix"].rstrip("/").split("/")[-1] for x in r.get("CommonPrefixes", [])]
    ranges[c] = names
allnames = collections.Counter()
for c, names in ranges.items():
    for n in names:
        allnames[n.split(".", 1)[1] if "." in n else n] += 1
    print(c, names)
print("distinct date ranges", len(allnames))
for k, v in sorted(allnames.items()):
    print(v, k)
