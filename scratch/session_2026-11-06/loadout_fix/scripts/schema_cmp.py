import io, struct, json
import boto3
import pyarrow.parquet as pq
from botocore.config import Config

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}))


def body(key, rng=None):
    kw = {"Bucket": BUCKET, "Key": key}
    if rng:
        kw["Range"] = rng
    return s3.get_object(**kw)["Body"].read()


def footer(key):
    size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    tail = body(key, f"bytes={size-8}-{size-1}")
    assert tail[4:] == b"PAR1", key
    flen = struct.unpack("<I", tail[:4])[0]
    start = size - 8 - flen
    part = body(key, f"bytes={start}-{size-1}")
    buf = b"PAR1" + b"\0" * (start - 4) + part
    md = pq.read_metadata(io.BytesIO(buf))
    sch = pq.ParquetFile(io.BytesIO(buf)).schema_arrow
    return md, sch, size


def folder_of(c, prefix):
    base = f"adam/tomes/chunks/{c}/tome/csds/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=base + prefix, Delimiter="/")
    folder = resp["CommonPrefixes"][0]["Prefix"]
    return folder


targets = {}
for c in ["c000", "c046", "c047", "c050", "c067"]:
    f = folder_of(c, "basic_player_status.")
    targets[c] = f + "dataframe_00000"
schemas = {}
for c, k in targets.items():
    md, sch, size = footer(k)
    schemas[c] = sch
    print(c, "rows", md.num_rows, "row_groups", md.num_row_groups, "size", size, "created_by", md.created_by)

names = list(schemas["c000"].names)
print("\nfull schema c000 (v3):")
for f in schemas["c000"]:
    print("  ", f.name, str(f.type))
print("\nschema diffs vs c000:")
for c in ["c046", "c047", "c050", "c067"]:
    s = schemas[c]
    same_names = s.names == schemas["c000"].names
    diffs = []
    for a, b in zip(schemas["c000"], s):
        if a.name != b.name or str(a.type) != str(b.type):
            diffs.append((a.name, str(a.type), b.name, str(b.type)))
    print(c, "same names and order:", same_names, "| n cols", len(s), "| diffs (c000 name,type | this name,type):", diffs)

# basic_player_info: look for a side/team column in another channel of the same chunk
base = "adam/tomes/chunks/c047/tome/csds/"
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=base + "basic_player_info.", Delimiter="/")
pi = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
print("\nplayer_info folders c047:", pi)
if pi:
    k = pi[0] + "dataframe_00000"
    md, sch, size = footer(k)
    print("basic_player_info cols:", [f.name for f in sch])
