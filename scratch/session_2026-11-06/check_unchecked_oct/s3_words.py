import boto3, re, json, collections, time
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config
B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
s3 = boto3.client("s3", config=Config(retries={"max_attempts": 8, "mode": "standard"}, max_pool_connections=32))
WORDS = re.compile(r"tome|chunk|combined|adam", re.I)
def list_all(prefix):
    out = []; tok = None
    while True:
        kw = dict(Bucket=B, Prefix=prefix, MaxKeys=1000)
        if tok: kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out += [(x["Key"], x["Size"]) for x in r.get("Contents", [])]
        if r.get("IsTruncated"): tok = r["NextContinuationToken"]
        else: break
    return prefix, out
t0 = time.time()
# day-level prefixes of data/csds for parallel recursive listing, plus other roots
roots = ["data/tomes/", "adam/", "data/csds/2023/", "data/csds/2024/"]
# data/csds/2026 months and days
def delim(p):
    out = []; tok = None
    while True:
        kw = dict(Bucket=B, Prefix=p, Delimiter="/", MaxKeys=1000)
        if tok: kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out += [x["Prefix"] for x in r.get("CommonPrefixes", [])]
        direct = [x["Key"] for x in r.get("Contents", [])]
        if r.get("IsTruncated"): tok = r["NextContinuationToken"]
        else: break
    return out, direct
for y in ["data/csds/2026/"]:
    for m in delim(y)[0]:
        roots.append(m)
# day level for Oct and Jan-Jul as direct listing via months
targets = []
for r in roots:
    targets.append(r)
# direct objects at year/month level for data/csds (non-uuid files), checked separately
direct_csds = {}
for y in ["data/csds/", "data/csds/2023/", "data/csds/2024/", "data/csds/2026/"]:
    cps, d = delim(y)
    direct_csds[y] = d[:20]
print("direct objects at data/csds levels:", {k: len(v) for k, v in direct_csds.items()}, flush=True)
results = {}
word_keys = []
sizes = collections.Counter()
per_top = collections.Counter()
with ThreadPoolExecutor(max_workers=24) as ex:
    for prefix, objs in ex.map(list_all, targets):
        results[prefix] = len(objs)
        for k, sz in objs:
            per_top[prefix] += 1
            sizes[prefix] += sz
            if WORDS.search(k):
                word_keys.append((k, sz))
print("listed prefixes", len(targets), "elapsed", round(time.time()-t0,1), flush=True)
for p in targets: print(p, "objects:", results[p], "bytes:", sizes[p], flush=True)
print("keys whose path contains tome/chunk/combined/adam:", len(word_keys), flush=True)
pref_counter = collections.Counter()
for k, sz in word_keys:
    parts = k.split("/")
    pref_counter["/".join(parts[:-1]) + "/"] += 1
for p, c in sorted(pref_counter.items())[:200]:
    print("  word-prefix", p, c)
json.dump({"targets": {t: results[t] for t in targets}, "direct_csds": direct_csds,
           "word_prefix_counts": dict(pref_counter), "word_key_samples": word_keys[:300]},
          open(W + "s3_words_summary.json", "w"), indent=1)
with open(W + "s3_word_keys.tsv", "w") as f:
    for k, sz in word_keys: f.write(f"{k}\t{sz}\n")
print("done", round(time.time()-t0,1), flush=True)
