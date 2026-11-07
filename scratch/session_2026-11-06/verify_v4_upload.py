"""Read every uploaded object back from the RunPod volume, hash it while streaming, and compare with the local source.

Nothing is written to disk. Output: upload_v4/verify.json and a printed summary.
"""
import configparser
import hashlib
import json
import os
import time

import boto3
from botocore.config import Config

BUCKET = "z7yvc40ykx"
ENDPOINT = "https://s3api-eu-ro-1.runpod.io"
SRC = "/home/sagemaker-user/adam/builder/final/export_clean_v4"
OUTD = "/home/sagemaker-user/adam/builder/fixes/readiness/upload_v4"

cp = configparser.ConfigParser()
cp.read(os.path.expanduser("~/.aws/runpod_s3_credentials"))
sec = cp["runpods3"]
s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    region_name="eu-ro-1",
    aws_access_key_id=sec["aws_access_key_id"].strip(),
    aws_secret_access_key=sec["aws_secret_access_key"].strip(),
    config=Config(signature_version="s3v4", request_checksum_calculation="when_required",
                  response_checksum_validation="when_required", retries={"max_attempts": 10, "mode": "adaptive"}),
)

files = [l.strip()[2:] for l in open(os.path.join(OUTD, "file_list.txt")) if l.strip()]
results = []
t0 = time.time()
total_bytes = 0
for rel in files:
    local = os.path.join(SRC, rel)
    hl = hashlib.sha256()
    with open(local, "rb") as f:
        for b in iter(lambda: f.read(64 * 1024 * 1024), b""):
            hl.update(b)
    n = 0
    hr = None
    for attempt in range(1, 6):
        try:
            hr = hashlib.sha256()
            n = 0
            obj = s3.get_object(Bucket=BUCKET, Key="export_clean_v4/" + rel)
            for chunk in obj["Body"].iter_chunks(64 * 1024 * 1024):
                hr.update(chunk)
                n += len(chunk)
            if n != obj["ContentLength"]:
                raise IOError(f"short read {n} of {obj['ContentLength']}")
            break
        except Exception as e:  # network drops mid-stream: retry the whole object
            print(f"retry {attempt} {rel}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(5 * attempt)
    if hr is None or n == 0:
        raise SystemExit(f"could not read {rel} after retries")
    total_bytes += n
    ok = hl.hexdigest() == hr.hexdigest() and n == os.path.getsize(local)
    results.append({"file": rel, "bytes": n, "local_sha256": hl.hexdigest(), "remote_sha256": hr.hexdigest(), "ok": ok})
    print(("OK   " if ok else "FAIL ") + rel + f" {n/1e6:.1f} MB", flush=True)

bad = [r["file"] for r in results if not r["ok"]]
summary = {"files": len(results), "ok": len(results) - len(bad), "failed": bad,
           "bytes_read_back": total_bytes, "seconds": round(time.time() - t0, 1), "results": results}
json.dump(summary, open(os.path.join(OUTD, "verify.json"), "w"), indent=1)
print("VERIFY_DONE files", len(results), "ok", len(results) - len(bad), "failed", len(bad), flush=True)
