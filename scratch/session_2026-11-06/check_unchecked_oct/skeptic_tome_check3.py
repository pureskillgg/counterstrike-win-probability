import boto3, csv, io, json, time, re, collections
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config

B = "pureskillgg-csgo-production-sagemaker-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
UU = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
targets_full = set()
target_uuids = set()
for r in csv.DictReader(open("/home/sagemaker-user/adam/builder/fixes/fix332/undetermined_matches.csv")):
    targets_full.add(r["match_key"])
    target_uuids.add(r["match_key"].split("/")[4])
cfg = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=64)

d = json.load(open(W + "skeptic_tome_check2.json"))
todo = []
for o in d["objects"]:
    if not o.get("parquet") or "distinct_match_keys" in o:
        continue
    cols = o.get("cols", [])
    if cols == ["_"]:
        todo.append((o["key"], o["size"], ["_"]))
    elif any(c in ("key", "match_id") for c in cols):
        todo.append((o["key"], o["size"], [c for c in cols if c in ("key", "match_id")]))
print("objects to read", len(todo), "bytes", sum(s for _, s, _ in todo), flush=True)


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


def check(item):
    k, size, cols = item
    s3 = boto3.client("s3", config=cfg)
    out = {"key": k, "size": size, "cols": cols}
    try:
        pf = pq.ParquetFile(io.BufferedReader(S3Range(s3, k, size), buffer_size=1 << 20))
        t = pf.read(columns=cols)
    except Exception as e:
        out["err"] = str(e)[:100]
        return out
    vals = []
    for c in cols:
        vals += [str(v) for v in t.column(c).to_pylist() if v is not None]
    out["values"] = len(vals)
    out["distinct"] = len(set(vals))
    full_hits = 0
    uuid_hits = 0
    uuids_seen = set()
    for v in set(vals):
        if v in targets_full:
            full_hits += 1
        for u in UU.findall(v):
            uuids_seen.add(u)
            if u in target_uuids:
                uuid_hits += 1
    out["target_full_hits"] = full_hits
    out["target_uuid_hits"] = uuid_hits
    out["uuids_seen"] = len(uuids_seen)
    out["sample"] = sorted(set(vals))[:1]
    return out


t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=24) as ex:
    futs = [ex.submit(check, it) for it in todo]
    for i, fut in enumerate(as_completed(futs)):
        results.append(fut.result())
        if i % 1000 == 0:
            print("progress", i, round(time.time() - t0, 1), flush=True)

summary = {
    "objects": len(results),
    "errors": sum(1 for r in results if "err" in r),
    "objects_with_values": sum(1 for r in results if r.get("values", 0) > 0),
    "values_total": sum(r.get("values", 0) for r in results),
    "target_full_hits_total": sum(r.get("target_full_hits", 0) for r in results),
    "target_uuid_hits_total": sum(r.get("target_uuid_hits", 0) for r in results),
    "objects_with_any_target_uuid": sum(1 for r in results if r.get("target_uuid_hits", 0) > 0),
    "uuids_seen_total_distinct_not_computed": True,
    "by_kind": {
        "single_underscore_col": collections.Counter(tuple(r["cols"]) for r in results).most_common(3),
    },
    "seconds": round(time.time() - t0, 1),
}
with open(W + "skeptic_tome_check3.json", "w") as f:
    json.dump({"summary": summary, "objects": sorted(results, key=lambda r: r["key"])}, f, indent=1, default=str)
print("SUMMARY", json.dumps(summary, default=str), flush=True)
print("TOME3_DONE", flush=True)
