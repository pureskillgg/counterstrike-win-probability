import boto3, sys, io, struct, os
import pyarrow.parquet as pq
from botocore.config import Config
B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}))
def footer_schema(key):
    head = s3.head_object(Bucket=B, Key=key)
    size = head["ContentLength"]
    tail = s3.get_object(Bucket=B, Key=key, Range=f"bytes={size-8}-{size-1}")["Body"].read()
    flen = struct.unpack("<I", tail[:4])[0]
    buf = s3.get_object(Bucket=B, Key=key, Range=f"bytes={size-8-flen}-{size-1}")["Body"].read()
    meta = pq.read_metadata(io.BytesIO(b"PAR1" + b"\0"*0 + buf)) if False else None
    # build a minimal file-like with footer only is not valid for pq; fall back to full small download
    return size
for k in sys.argv[1:]:
    h = s3.head_object(Bucket=B, Key=k)
    print("==", k, "size", h["ContentLength"], flush=True)
