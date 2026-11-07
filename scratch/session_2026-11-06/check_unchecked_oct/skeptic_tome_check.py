import boto3, csv, io, json, time, re, collections
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
s3 = boto3.client("s3", config=cfg)

# list every object under data/tomes/ and adam/tomes/, excluding the holdout folder (holdout rule)
objs = []
for pfx in ["data/tomes/", "adam/tomes/"]:
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=B, Prefix=pfx):
        for o in page.get("Contents", []):
            k = o["Key"]
            if "holdout_2026-10-26_31" in k:
                continue
            objs.append((k, o["Size"]))
print("objects to read", len(objs), "bytes", sum(s for _, s in objs), flush=True)

# also check for any objects directly under the bucket root or data/ (files, not folders)
for pfx in ["", "data/"]:
    r = s3.list_objects_v2(Bucket=B, Prefix=pfx, Delimiter="/", MaxKeys=1000)
    print("direct files under", repr(pfx), len(r.get("Contents", [])), flush=True)

def check(item):
    k, size = item
    out = {"key": k, "size": size}
    if size > 400_000_000:
        out["skipped_large"] = True
        return out
    try:
        body = s3.get_object(Bucket=B, Key=k)["Body"].read()
    except Exception as e:
        out["error"] = "get:" + str(e)[:100]
        return out
    try:
        sch = pq.read_schema(io.BytesIO(body))
        names = sch.names
    except Exception:
        out["parquet"] = False
        return out
    out["parquet"] = True
    out["cols"] = names
    if "match_key" not in names:
        return out
    t = pq.read_table(io.BytesIO(body), columns=["match_key"])
    vals = set(v for v in t.column("match_key").to_pylist() if v is not None)
    out["rows"] = t.num_rows
    out["distinct_match_keys"] = len(vals)
    out["match_key_sample"] = sorted(vals)[:1]
    out["target_full_hits"] = sum(1 for v in vals if v in targets_full)
    out["target_uuid_hits"] = sum(1 for v in vals if v.split("/")[4] in target_uuids) if vals else 0
    out["_vals"] = list(vals) if ("basic_player_info" in k) else None
    return out

t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(check, it) for it in objs]
    for i, fut in enumerate(as_completed(futs)):
        results.append(fut.result())
        if i % 1000 == 0:
            print("progress", i, round(time.time() - t0, 1), flush=True)

bpi_keys_all = set()
bpi_objs = 0
for r in results:
    if r.get("_vals"):
        bpi_keys_all |= set(r["_vals"])
        bpi_objs += 1
    r.pop("_vals", None)

# compare basic_player_info distinct match keys with the S3 folder list (uuid-level)
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
    "objects_read": len(results),
    "parquet_objects": sum(1 for r in results if r.get("parquet")),
    "objects_with_match_key": sum(1 for r in results if "distinct_match_keys" in r),
    "objects_with_team_code": sum(1 for r in results if "cols" in r and "team_code" in r["cols"]),
    "basic_player_info_objects_with_match_key": bpi_objs,
    "basic_player_info_distinct_match_keys": len(bpi_keys_all),
    "target_full_hits_all": sum(r.get("target_full_hits", 0) for r in results),
    "target_uuid_hits_all": sum(r.get("target_uuid_hits", 0) for r in results),
    "bpi_uuids_in_targets": len(bpi_uuids & target_uuids),
    "bpi_uuids_in_s3_folders": len(bpi_uuids & s3_folder_uuids),
    "bpi_uuids_not_in_s3_folders": len(bpi_uuids - s3_folder_uuids),
    "s3_oct_folder_uuids": len(s3_oct_folder_uuids),
    "s3_oct_folder_uuids_in_bpi": len(s3_oct_folder_uuids & bpi_uuids),
    "seconds": round(time.time() - t0, 1),
    "errors": sum(1 for r in results if "error" in r),
}
with open(W + "skeptic_tome_check.json", "w") as f:
    json.dump({"summary": summary, "objects": sorted(results, key=lambda r: r["key"])}, f, indent=1)
print("SUMMARY", json.dumps(summary), flush=True)
print("TOME_DONE", flush=True)
