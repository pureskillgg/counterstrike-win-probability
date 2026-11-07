import boto3, csv, io, json, time
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config

B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
targets_full = set()
target_uuids = set()
with open("/home/sagemaker-user/adam/builder/fixes/fix332/undetermined_matches.csv") as f:
    for r in csv.DictReader(f):
        targets_full.add(r["match_key"])
        target_uuids.add(r["match_key"].split("/")[4])
print("targets", len(targets_full), flush=True)

cfg = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=64)


class S3Range(io.RawIOBase):
    def __init__(self, s3, key, size):
        self.s3 = s3
        self.key = key
        self.size = size
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buf):
        n = len(buf)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n, self.size) - 1
        data = self.s3.get_object(Bucket=B, Key=self.key, Range=f"bytes={self.pos}-{end}")["Body"].read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


# list objects under data/tomes/ and adam/tomes/, excluding holdout folder (holdout rule)
lister = boto3.client("s3", config=cfg)
objs = []
for pfx in ["data/tomes/", "adam/tomes/"]:
    pag = lister.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=B, Prefix=pfx):
        for o in page.get("Contents", []):
            if "holdout_2026-10-26_31" in o["Key"]:
                continue
            objs.append((o["Key"], o["Size"]))
print("objects to check", len(objs), "bytes", sum(s for _, s in objs), flush=True)


def check(item):
    k, size = item
    s3 = boto3.client("s3", config=cfg)
    out = {"key": k, "size": size}
    raw = S3Range(s3, k, size)
    bf = io.BufferedReader(raw, buffer_size=1 << 20)
    try:
        pf = pq.ParquetFile(bf)
        names = pf.schema_arrow.names
        out["parquet"] = True
        out["cols"] = names
        out["num_rows"] = pf.metadata.num_rows
    except Exception as e:
        out["parquet"] = False
        out["err"] = str(e)[:80]
        return out
    if "match_key" not in names:
        return out
    try:
        t = pf.read(columns=["match_key"])
    except Exception as e:
        out["read_err"] = str(e)[:80]
        return out
    vals = set(v for v in t.column("match_key").to_pylist() if v is not None)
    out["distinct_match_keys"] = len(vals)
    out["target_full_hits"] = sum(1 for v in vals if v in targets_full)
    out["target_uuid_hits"] = sum(1 for v in vals if v.split("/")[4] in target_uuids)
    out["_vals"] = sorted(vals) if "basic_player_info" in k else None
    return out


t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=24) as ex:
    futs = [ex.submit(check, it) for it in objs]
    for i, fut in enumerate(as_completed(futs)):
        results.append(fut.result())
        if i % 1000 == 0:
            print("progress", i, round(time.time() - t0, 1), flush=True)
print("all checked", round(time.time() - t0, 1), flush=True)

bpi_keys_all = set()
bpi_objs = 0
for r in results:
    if r.get("_vals"):
        bpi_keys_all |= set(r["_vals"])
        bpi_objs += 1
    r.pop("_vals", None)

s3_folder_uuids = set()
s3_oct_folder_uuids = set()
with open(W + "skeptic_s3_folders.tsv") as f:
    next(f)
    for line in f:
        y, m, d, u = line.rstrip("\n").split("\t")
        s3_folder_uuids.add(u)
        if y == "2026" and m == "10":
            s3_oct_folder_uuids.add(u)
bpi_uuids = set(k.split("/")[4] for k in bpi_keys_all if k.count("/") >= 4)
summary = {
    "objects_checked": len(results),
    "parquet_objects": sum(1 for r in results if r.get("parquet")),
    "non_parquet_objects": sum(1 for r in results if r.get("parquet") is False),
    "objects_with_match_key": sum(1 for r in results if "distinct_match_keys" in r),
    "objects_with_team_code": sum(1 for r in results if "team_code" in r.get("cols", [])),
    "basic_player_info_objects_with_match_key": bpi_objs,
    "basic_player_info_distinct_match_keys": len(bpi_keys_all),
    "target_full_hits_all": sum(r.get("target_full_hits", 0) for r in results),
    "target_uuid_hits_all": sum(r.get("target_uuid_hits", 0) for r in results),
    "bpi_uuids_in_targets": len(bpi_uuids & target_uuids),
    "bpi_uuids_in_s3_folders": len(bpi_uuids & s3_folder_uuids),
    "bpi_uuids_not_in_s3_folders": len(bpi_uuids - s3_folder_uuids),
    "s3_oct_folder_uuids": len(s3_oct_folder_uuids),
    "s3_oct_folder_uuids_in_bpi": len(s3_oct_folder_uuids & bpi_uuids),
    "errors": sum(1 for r in results if "err" in r or "read_err" in r),
    "seconds": round(time.time() - t0, 1),
}
with open(W + "skeptic_tome_check2.json", "w") as f:
    json.dump({"summary": summary, "objects": sorted(results, key=lambda r: r["key"])}, f, indent=1)
print("SUMMARY", json.dumps(summary), flush=True)
print("TOME_DONE", flush=True)
