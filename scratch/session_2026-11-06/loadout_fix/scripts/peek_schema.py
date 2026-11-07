import io, struct, json
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 5}))
P = "adam/tomes/chunks/c000/tome/csds/basic_player_status.2023-11-10,2023-11-26/"
# keyset (key listing) schema, whole small file
kb = s3.get_object(Bucket=BUCKET, Key=P + "keyset_00000")["Body"].read()
print("keyset_00000 bytes", len(kb), "magic", kb[:4], kb[-4:])
kt = pq.read_table(io.BytesIO(kb))
print("keyset schema:", kt.schema.to_string())
print("keyset rows", kt.num_rows)
print("keyset first 3 rows:", [ {c: str(kt.column(c)[i].as_py())[:160] for c in kt.column_names} for i in range(min(3, kt.num_rows))])
# dataframe footer only, by byte range
def footer_schema(key):
    size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    tail = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={size-8}-{size-1}")["Body"].read()
    assert tail[4:] == b"PAR1", tail
    flen = struct.unpack("<I", tail[:4])[0]
    start = size - 8 - flen
    body = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={start}-{size-1}")["Body"].read()
    buf = b"PAR1" + b"\0" * (start - 4) + body  # zero-filled gap; only the footer is read
    md = pq.read_metadata(io.BytesIO(buf))
    sch = pq.ParquetFile(io.BytesIO(buf)).schema_arrow
    return size, flen, md, sch
size, flen, md, sch = footer_schema(P + "dataframe_00000")
print("dataframe_00000 size", size, "footer_len", flen, "num_rows", md.num_rows, "row_groups", md.num_row_groups, "created_by", md.created_by)
print("columns", len(sch.names))
for f in sch:
    print(f"  {f.name}: {f.type}")
