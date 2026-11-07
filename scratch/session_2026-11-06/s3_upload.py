"""Upload files to the RunPod network volume through its S3 API (multipart, parallel).

Usage: python s3_upload.py <local_file> <key> [<local_file> <key> ...]
Credentials: ~/.aws/runpod_s3_credentials (profile runpods3). Never printed.
"""
import configparser
import hashlib
import os
import sys
import time

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

BUCKET = "z7yvc40ykx"
ENDPOINT = "https://s3api-eu-ro-1.runpod.io"
REGION = "eu-ro-1"
CREDS = os.path.expanduser("~/.aws/runpod_s3_credentials")

cp = configparser.ConfigParser()
cp.read(CREDS)
sec = cp["runpods3"]

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    region_name=REGION,
    aws_access_key_id=sec["aws_access_key_id"].strip(),
    aws_secret_access_key=sec["aws_secret_access_key"].strip(),
    config=Config(
        signature_version="s3v4",
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        retries={"max_attempts": 10, "mode": "adaptive"},
        max_pool_connections=32,
    ),
)

TC = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,
    multipart_chunksize=100 * 1024 * 1024,
    max_concurrency=int(os.environ.get("MAXC", "16")),
    use_threads=True,
)


def sha256_file(path, block=64 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(pairs):
    for local, key in zip(pairs[::2], pairs[1::2]):
        size = os.path.getsize(local)
        t0 = time.time()
        s3.upload_file(local, BUCKET, key, Config=TC)
        dt = time.time() - t0
        head = s3.head_object(Bucket=BUCKET, Key=key)
        ok = head["ContentLength"] == size
        print(f"{key}: {size/1e9:.3f} GB in {dt:.1f} s = {size/1e6/max(dt,1e-9):.1f} MB/s; remote size match={ok}", flush=True)
        if not ok:
            sys.exit(f"size mismatch for {key}")


if __name__ == "__main__":
    main(sys.argv[1:])
