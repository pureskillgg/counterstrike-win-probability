import boto3, csv, io, re, json, collections, time
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config
B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}, max_pool_connections=32))
keys = [r["match_key"] for r in csv.DictReader(open("/home/sagemaker-user/adam/builder/fixes/fix332/undetermined_matches.csv"))]
KS = set(keys)
rows = [l.rstrip("\n").split("\t") for l in open(W + "s3_word_keys.tsv")]
keysets = [k for k, s in rows if "/keyset_" in k and (k.startswith("adam/tomes/chunks/") or k.startswith("data/tomes/"))]
bpi = [k for k, s in rows if "basic_player_info" in k and "/dataframe_" in k and (k.startswith("adam/tomes/chunks/") or k.startswith("data/tomes/"))]
print("keyset objects", len(keysets), "basic_player_info dataframe objects", len(bpi), flush=True)
def src_of(k):
    m = re.match(r"adam/tomes/chunks/(c\d+)/", k)
    if m: return m.group(1)
    m = re.match(r"data/tomes/tome/csds/([^/]+)/", k)
    return "data/" + (m.group(1) if m else "?")
def read_keys(k):
    body = s3.get_object(Bucket=B, Key=k)["Body"].read()
    t = pq.read_table(io.BytesIO(body), columns=["_"]) if True else None
    return k, t.column("_").to_pylist()
t0 = time.time()
ks_cov = collections.defaultdict(set); ks_all = collections.defaultdict(set)
ks_err = []
with ThreadPoolExecutor(max_workers=24) as ex:
    futs = {ex.submit(read_keys, k): k for k in keysets}
    for f in futs:
        try:
            k, vals = f.result()
        except Exception as e:
            ks_err.append((futs[f], str(e)[:120])); continue
        s = src_of(k)
        ks_all[s].update(vals)
        ks_cov[s].update(v for v in vals if v in KS)
print("keyset read done", round(time.time()-t0,1), "errors", len(ks_err), flush=True)
allk = set().union(*ks_cov.values()) if ks_cov else set()
print("KEYSET union covering target keys:", len(allk), flush=True)
for s in sorted(ks_cov):
    print("  keyset source", s, "rows", len(ks_all[s]), "target-hits", len(ks_cov[s]), flush=True)
# basic_player_info dataframes: match_key column + team_code presence
def read_bpi(k):
    body = s3.get_object(Bucket=B, Key=k)["Body"].read()
    sch = pq.read_schema(io.BytesIO(body))
    names = sch.names
    cols = ["match_key"] if "match_key" in names else []
    if "team_code" in names: cols.append("team_code")
    t = pq.read_table(io.BytesIO(body), columns=cols) if cols else None
    mk = t.column("match_key").to_pylist() if t is not None and "match_key" in cols else []
    return k, names, mk, len(body)
bpi_cov = collections.defaultdict(set); bpi_all = collections.defaultdict(set)
bpi_schema = {}; bpi_err = []; bpi_rows = collections.Counter()
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = {ex.submit(read_bpi, k): k for k in bpi}
    for f in futs:
        try:
            k, names, mk, nbytes = f.result()
        except Exception as e:
            bpi_err.append((futs[f], str(e)[:160])); continue
        s = src_of(k)
        bpi_schema.setdefault(s, names)
        bpi_all[s].update(mk)
        bpi_rows[s] += len(mk)
        bpi_cov[s].update(v for v in mk if v in KS)
print("basic_player_info read done", round(time.time()-t0,1), "errors", len(bpi_err), flush=True)
for e in bpi_err[:10]: print("  ERR", e, flush=True)
allb = set().union(*bpi_cov.values()) if bpi_cov else set()
print("BPI union covering target keys:", len(allb), flush=True)
for s in sorted(bpi_all):
    print("  bpi source", s, "rows", bpi_rows[s], "distinct matches", len(bpi_all[s]), "target-hits", len(bpi_cov[s]), "team_code" in bpi_schema.get(s, []), flush=True)
json.dump({"keysets": len(keysets), "keyset_errors": ks_err[:20], "keyset_union_hits": len(allk),
           "keyset_hits_by_source": {s: len(v) for s, v in ks_cov.items()},
           "keyset_rows_by_source": {s: len(v) for s, v in ks_all.items()},
           "bpi_files": len(bpi), "bpi_errors": bpi_err[:20], "bpi_union_hits": len(allb),
           "bpi_hits_by_source": {s: len(v) for s, v in bpi_cov.items()},
           "bpi_distinct_matches_by_source": {s: len(v) for s, v in bpi_all.items()},
           "bpi_schema_by_source": {s: v for s, v in bpi_schema.items()},
           "sample_hits": sorted(allk | allb)[:10]},
          open(W + "s3_tome_coverage.json", "w"), indent=1)
print("WROTE", round(time.time()-t0,1), flush=True)
