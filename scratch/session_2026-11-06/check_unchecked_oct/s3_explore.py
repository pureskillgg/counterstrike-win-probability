import boto3, sys
from botocore.config import Config
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 5, "mode": "standard"}))
B = "pureskillgg-csgo-production-sagemaker-0"

def pref(p, d="/"):
    out = []; tok = None
    while True:
        kw = dict(Bucket=B, Prefix=p, Delimiter=d, MaxKeys=1000)
        if tok: kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out += [x["Prefix"] for x in r.get("CommonPrefixes", [])]
        out_files = [x["Key"] for x in r.get("Contents", [])]
        if r.get("IsTruncated"): tok = r["NextContinuationToken"]
        else: break
    return out, out_files

for p in sys.argv[1:]:
    cps, files = pref(p)
    print(f"PREFIX {p!r}: {len(cps)} common prefixes, {len(files)} direct objects")
    print("  sample:", cps[:40])
    if files: print("  files:", files[:20])
