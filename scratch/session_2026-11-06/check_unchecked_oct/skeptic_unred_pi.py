import boto3, json, csv, io, hashlib, time, collections
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config

B = "pureskillgg-csgo-production-dataexchange-csds-unredacted-0"
W = "/home/sagemaker-user/adam/builder/fixes/check_unchecked_oct/"
cfg = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=64)
items = []
for r in csv.DictReader(open("/home/sagemaker-user/adam/builder/fixes/fix332/undetermined_matches.csv")):
    items.append(r["match_key"])


def one(mk):
    s3 = boto3.client("s3", config=cfg)
    key = mk[:-len("csds")] + "player_info"
    body = s3.get_object(Bucket=B, Key=key)["Body"].read()
    t = pq.read_table(io.BytesIO(body))
    names = t.column_names
    out = {"key": mk, "cols": names, "rows": t.num_rows}
    if "team_code" in names and "round" in names and "player_id" in names:
        rows = sorted([str(a), str(b), str(c)] for a, b, c in zip(
            t.column("round").to_pylist(), t.column("player_id").to_pylist(), t.column("team_code").to_pylist()))
        out["md5"] = hashlib.md5(json.dumps(rows).encode()).hexdigest()
        out["tc_counts"] = dict(collections.Counter(str(v) for v in t.column("team_code").to_pylist()))
    return out


t0 = time.time()
with ThreadPoolExecutor(max_workers=32) as ex:
    res = list(ex.map(one, items))
with open(W + "skeptic_unred_pi_hashes.json", "w") as f:
    json.dump(res, f)
print("n", len(res), "with team_code", sum(1 for r in res if "md5" in r), "secs", round(time.time() - t0, 1), flush=True)
print("col sets", collections.Counter(tuple(r["cols"]) for r in res).most_common(2), flush=True)
print("PI_DONE", flush=True)
