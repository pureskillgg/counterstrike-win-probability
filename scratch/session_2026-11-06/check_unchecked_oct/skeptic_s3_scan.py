import boto3, re, csv, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config

B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
FOLDER_RE = re.compile(r"^data/csds/(\d{4})/(\d{2})/(\d{2})/([0-9a-f-]{36})/")

targets = {}
with open("/home/sagemaker-user/adam/builder/fixes/fix332/undetermined_matches.csv") as f:
    for r in csv.DictReader(f):
        mk = r["match_key"]
        targets[mk.split("/")[4]] = mk
print("targets", len(targets), flush=True)

cfg = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=64)

units = []
for y in ["2023", "2024"]:
    for m in range(1, 13):
        units.append(f"data/csds/{y}/{m:02d}/")
for m in range(1, 13):
    if m != 10:
        units.append(f"data/csds/2026/{m:02d}/")
for d in range(1, 32):
    units.append(f"data/csds/2026/10/{d:02d}/")
units += ["data/tomes/", "adam/", "data/csds/", "data/"]
# data/csds/ and data/ above are catch-alls for keys not under the units; they re-list
# everything, so they are only used for a stray-key check on the top-level prefixes.
units = [u for u in units if u not in ("data/csds/", "data/")]


def scan(prefix):
    s3 = boto3.client("s3", config=cfg)
    pag = s3.get_paginator("list_objects_v2")
    n = 0
    any_uuids = set()
    folders = set()
    hits = []
    for page in pag.paginate(Bucket=B, Prefix=prefix, PaginationConfig={"PageSize": 1000}):
        for o in page.get("Contents", []):
            k = o["Key"]
            n += 1
            for u in UUID_RE.findall(k):
                any_uuids.add(u)
                if u in targets:
                    hits.append(k)
            m = FOLDER_RE.match(k)
            if m:
                folders.add((m.group(1), m.group(2), m.group(3), m.group(4)))
    return {"prefix": prefix, "keys": n, "any_uuids": any_uuids, "folders": folders, "hits": hits}


t0 = time.time()
agg_keys = 0
all_any = set()
all_folders = set()
all_hits = []
per_unit = {}
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(scan, u): u for u in units}
    for fut in as_completed(futs):
        r = fut.result()
        agg_keys += r["keys"]
        all_any |= r["any_uuids"]
        all_folders |= r["folders"]
        all_hits += r["hits"]
        per_unit[r["prefix"]] = {"keys": r["keys"], "folders": len(r["folders"]), "target_hits": len(r["hits"])}
        print("done", r["prefix"], r["keys"], len(r["folders"]), len(r["hits"]), round(time.time() - t0, 1), flush=True)

target_found = sorted(set(u for u in all_any if u in targets))
folders_by_year = {}
for y, m, d, u in all_folders:
    folders_by_year[f"{y}/{m}"] = folders_by_year.get(f"{y}/{m}", 0) + 1

with open(W + "skeptic_s3_folders.tsv", "w") as f:
    f.write("y\tm\td\tuuid\n")
    for y, m, d, u in sorted(all_folders):
        f.write(f"{y}\t{m}\t{d}\t{u}\n")
with open(W + "skeptic_s3_target_hits.txt", "w") as f:
    for k in all_hits:
        f.write(k + "\n")

summary = {
    "bucket": B,
    "targets": len(targets),
    "keys_scanned": agg_keys,
    "distinct_uuid_strings_in_any_key": len(all_any),
    "targets_found_in_any_key": len(target_found),
    "target_hit_keys": len(all_hits),
    "csds_uuid_folders_total": len(all_folders),
    "csds_uuid_folders_by_year_month": dict(sorted(folders_by_year.items())),
    "per_unit": dict(sorted(per_unit.items())),
    "seconds": round(time.time() - t0, 1),
}
with open(W + "skeptic_s3_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "per_unit"}), flush=True)
print("FULLSCAN_DONE", flush=True)
