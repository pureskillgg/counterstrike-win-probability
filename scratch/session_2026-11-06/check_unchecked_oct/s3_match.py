import boto3, csv, collections, json, re
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config
B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
IN = "/home/sagemaker-user/adam/builder/fixes/fix332/undetermined_matches.csv"
rows = list(csv.DictReader(open(IN)))
keys = [r["match_key"] for r in rows]
print("keys", len(keys), "unique", len(set(keys)))
def parse(k):
    p = k.split("/")  # csds/YYYY/MM/DD/uuid/csds
    return p[1], p[2], p[3], p[4]
kinfo = {k: parse(k) for k in keys}
uuid_to_keys = collections.defaultdict(list)
for k, (y, m, d, u) in kinfo.items(): uuid_to_keys[u].append(k)
print("distinct uuids in keys", len(uuid_to_keys))
idx = {}
idx_by_uuid = collections.defaultdict(list)
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
with open(W + "s3_uuid_index.tsv") as f:
    next(f)
    for line in f:
        d, up = line.rstrip("\n").split("\t")
        u = up.rstrip("/").split("/")[-1]
        idx_by_uuid[u].append(up)
print("index uuids", len(idx_by_uuid), "index folders", sum(len(v) for v in idx_by_uuid.values()))
exact = []; anydate = []; date_mismatch = []
for k, (y, m, d, u) in kinfo.items():
    if u in idx_by_uuid:
        anydate.append(k)
        if any(up == f"data/csds/{y}/{m}/{d}/{u}/" for up in idx_by_uuid[u]): exact.append(k)
        else: date_mismatch.append((k, idx_by_uuid[u]))
print("keys whose uuid exists anywhere in data/csds:", len(anydate))
print("keys with exact date folder in data/csds:", len(exact))
print("uuid present but different date:", len(date_mismatch), date_mismatch[:5])
bymonth = collections.Counter()
for k in keys:
    y, m, d, u = kinfo[k]
    bymonth[(f"{y}/{m}", u in idx_by_uuid)] += 1
print("by month (month, uuid_present):", dict(bymonth))
# list objects inside matched folders (names only, no GET)
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}, max_pool_connections=32))
targets = sorted({up for k in anydate for up in idx_by_uuid[kinfo[k][3]]})
def names(prefix):
    out = []; tok = None
    while True:
        kw = dict(Bucket=B, Prefix=prefix, MaxKeys=1000)
        if tok: kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out += [x["Key"] for x in r.get("Contents", [])]
        if r.get("IsTruncated"): tok = r["NextContinuationToken"]
        else: break
    return prefix, out
roster_hits = {}
file_counter = collections.Counter()
with ThreadPoolExecutor(max_workers=24) as ex:
    for prefix, objs in ex.map(names, targets):
        roster_hits[prefix] = objs
        for o in objs:
            file_counter[o.split("/")[-1]] += 1
print("matched folders listed", len(targets), "objects", sum(len(v) for v in roster_hits.values()))
print("file name counts inside matched folders:", file_counter.most_common(30))
json.dump({"keys": len(keys), "anydate": len(anydate), "exact": len(exact),
           "date_mismatch": len(date_mismatch), "sample_anydate": anydate[:5],
           "targets": targets[:20], "roster_hits": {k: v[:50] for k, v in list(roster_hits.items())[:50]}},
          open(W + "s3_match_summary.json", "w"), indent=1)
print("WROTE summary")
