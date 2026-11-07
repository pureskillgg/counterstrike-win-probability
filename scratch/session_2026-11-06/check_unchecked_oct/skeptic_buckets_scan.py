import boto3, csv, json, re, time, sys, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config

W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
FOLDER_RE = re.compile(r"^csds/(\d{4})/(\d{2})/(\d{2})/([0-9a-f-]{36})[/.]")
targets = {}
for r in csv.DictReader(open("/home/sagemaker-user/adam/builder/fixes/fix332/undetermined_matches.csv")):
    targets[r["match_key"].split("/")[4]] = r["match_key"]
print("targets", len(targets), flush=True)

BUCKETS = sys.argv[1].split(",")
cfg = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=64)


def list_top(s3, b):
    prefixes = []
    root = 0
    tok = None
    while True:
        kw = dict(Bucket=b, Delimiter="/", MaxKeys=1000)
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        prefixes += [x["Prefix"] for x in r.get("CommonPrefixes", [])]
        root += len(r.get("Contents", []))
        if r.get("IsTruncated"):
            tok = r["NextContinuationToken"]
        else:
            break
    return prefixes, root


def scan_prefix(b, pfx):
    s3 = boto3.client("s3", config=cfg)
    n = 0
    hits = []
    folders = set()
    ym = collections.Counter()
    any_uuid_seen = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=pfx):
        for o in page.get("Contents", []):
            k = o["Key"]
            n += 1
            for u in UUID_RE.findall(k):
                any_uuid_seen.add(u)
                if u in targets:
                    hits.append(k)
            m = FOLDER_RE.match(k)
            if m:
                folders.add((m.group(1), m.group(2), m.group(3), m.group(4)))
    return {"pfx": pfx, "keys": n, "hits": hits, "folders": folders, "uuids": any_uuid_seen}


summary_all = {}
for b in BUCKETS:
    t0 = time.time()
    s3 = boto3.client("s3", config=cfg)
    prefixes, root_n = list_top(s3, b)
    if not prefixes:
        prefixes = [""]
    agg = {"keys": root_n, "hits": [], "folders": set(), "uuids": set()}
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(scan_prefix, b, p) for p in prefixes]
        for fut in as_completed(futs):
            r = fut.result()
            agg["keys"] += r["keys"]
            agg["hits"] += r["hits"]
            agg["folders"] |= r["folders"]
            agg["uuids"] |= r["uuids"]
    tgt_found = sorted(set(u for u in agg["uuids"] if u in targets))
    ym = collections.Counter(f"{y}/{m}" for y, m, d, u in agg["folders"])
    summary = {
        "bucket": b,
        "top_prefixes": len(prefixes),
        "keys": agg["keys"],
        "distinct_uuid_strings_in_keys": len(agg["uuids"]),
        "targets_found_in_key_names": len(tgt_found),
        "target_hit_keys": len(agg["hits"]),
        "hit_examples": agg["hits"][:5],
        "uuid_folders_by_year_month": dict(sorted(ym.items())[-12:]),
        "seconds": round(time.time() - t0, 1),
    }
    summary_all[b] = summary
    print("BUCKET", json.dumps(summary), flush=True)
    with open(W + "skeptic_bucket_" + b + ".json", "w") as f:
        json.dump(summary, f, indent=1)
print("BUCKETS_DONE", flush=True)
