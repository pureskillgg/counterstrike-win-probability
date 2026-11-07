import boto3, csv, io, re, json, collections
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config

B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}, max_pool_connections=32))
KS = {r["match_key"] for r in csv.DictReader(open("/home/sagemaker-user/adam/builder/fixes/fix332/undetermined_matches.csv"))}
rows = [l.rstrip("\n").split("\t") for l in open(W + "s3_word_keys.tsv")]
other = [(k, int(s)) for k, s in rows if k.startswith("adam/tomes/") and not k.startswith("adam/tomes/chunks/") and not k.startswith("adam/tomes/holdout")]
hold_keysets = [k for k, s in rows if k.startswith("adam/tomes/holdout") and "/keyset_" in k]
print("other adam/tomes objects", len(other), "holdout keyset objects", len(hold_keysets), flush=True)
grp = collections.Counter(k.split("/")[2] + "/" + k.split("/")[3] for k, _ in other)
print("groups", dict(grp), flush=True)

def inspect(item):
    k, sz = item
    body = s3.get_object(Bucket=B, Key=k)["Body"].read()
    try:
        sch = pq.read_schema(io.BytesIO(body)).names
    except Exception as e:
        return k, None, [], 0, "noparquet:" + str(e)[:60]
    cols = [c for c in sch if c in ("match_key", "team_code")]
    if "match_key" in sch:
        t = pq.read_table(io.BytesIO(body), columns=["match_key"])
        mk = set(v for v in t.column("match_key").to_pylist() if v is not None)
        hit = len(mk & KS)
    else:
        hit = 0
    return k, sch[:40], cols, hit, None

res = []
with ThreadPoolExecutor(max_workers=16) as ex:
    for r in ex.map(inspect, other):
        res.append(r)
summary = collections.defaultdict(lambda: {"files": 0, "target_hits": 0, "team_code_files": 0, "errors": 0})
for k, sch, cols, hit, err in res:
    g = k.split("/")[2] + "/" + k.split("/")[3]
    summary[g]["files"] += 1
    summary[g]["target_hits"] = max(summary[g]["target_hits"], hit)
    if "team_code" in cols: summary[g]["team_code_files"] += 1
    if err: summary[g]["errors"] += 1
print(json.dumps(summary, indent=1), flush=True)
# holdout: keyset match-key lists only (no roster or prediction content read)
hits = collections.Counter(); total = 0
def read_ks(k):
    body = s3.get_object(Bucket=B, Key=k)["Body"].read()
    t = pq.read_table(io.BytesIO(body))
    col = t.column(0).to_pylist()
    return k, col
with ThreadPoolExecutor(max_workers=16) as ex:
    hk = list(ex.map(read_ks, hold_keysets))
hold_hits = set(); hold_rows = 0
for k, col in hk:
    hold_rows += len(col)
    hold_hits.update(v for v in col if v in KS)
print("holdout keyset rows", hold_rows, "target hits", len(hold_hits), flush=True)
json.dump({"other_groups": summary, "holdout_keyset_rows": hold_rows, "holdout_keyset_target_hits": len(hold_hits),
           "other_sample": [k for k, _, _, _, _ in res[:12]]},
          open(W + "s3_other_tomes.json", "w"), indent=1)
print("WROTE", flush=True)
