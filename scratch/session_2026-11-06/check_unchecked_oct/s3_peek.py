import boto3, sys, io
from botocore.config import Config
B = "pureskillgg-csgo-production-sagemaker-0"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}))
for k in sys.argv[1:]:
    try:
        r = s3.get_object(Bucket=B, Key=k, Range="bytes=0-500")
        body = r["Body"].read()
        print("==", k, "| head bytes:", body[:400], flush=True)
    except Exception as e:
        print("==", k, "| ERR", type(e).__name__, str(e)[:120], flush=True)
