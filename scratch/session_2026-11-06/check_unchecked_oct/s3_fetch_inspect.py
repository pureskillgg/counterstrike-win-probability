import boto3, sys, os
import pyarrow.parquet as pq
from botocore.config import Config
B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/tmp_tome/"
os.makedirs(W, exist_ok=True)
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}))
for k in sys.argv[1:]:
    local = W + k.replace("/", "__")
    if not os.path.exists(local):
        s3.download_file(B, k, local)
    t = pq.read_table(local)
    print("==", k, "rows", t.num_rows, "cols", t.num_columns, flush=True)
    print("   schema:", [(f.name, str(f.type)) for f in t.schema][:60], flush=True)
    df = t.slice(0, 3).to_pylist()
    for r in df[:2]:
        print("   sample row:", {kk: (str(vv)[:80]) for kk, vv in list(r.items())[:20]}, flush=True)
