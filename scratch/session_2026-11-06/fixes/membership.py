# Key-membership scan: which chunks c000-c067 hold each export key.
# Reads ONLY the match_key column of basic_player_status and basic_player_info pages. No feature or label column is read.
import boto3, io, json, sys, time
import polars as pl
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, file=sys.stderr, flush=True)

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
V3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
OUTD = "/home/sagemaker-user/adam/builder/fixes/loadout_chunk_locate/"
CHUNKS = ["c%03d" % i for i in range(0, 68)]

s3 = boto3.client("s3", config=Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "standard"}))
POOL = ThreadPoolExecutor(12)

keys = json.load(open(EXP + "match_keys.json"))
keyset = set(keys)
st_csv = pl.read_csv(V3 + "match_loadout_status.csv", schema_overrides={"match_key": pl.String})
assert st_csv["match_key"].to_list() == keys, "status csv order differs from match_keys.json"
log("export keys", len(keys), "status rows", st_csv.height)


def list_pages(c, kind):
    prefix = f"{PREFIX}/{c}/tome/csds/{kind}."
    out, folders = [], set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            d, n = o["Key"].rsplit("/", 1)
            if d.rsplit("/", 1)[-1].startswith(kind + ".") and n.startswith("dataframe_"):
                folders.add(d)
                out.append(o["Key"])
    return sorted(out), folders


def keys_in_pages(pkeys):
    def one(k):
        b = s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()
        return pl.read_parquet(io.BytesIO(b), columns=["match_key"])["match_key"].unique().to_list()
    res = set()
    for lst in POOL.map(one, pkeys):
        res.update(lst)
    return res


rows = []
per_chunk = {}
for c in CHUNKS:
    sp, sf = list_pages(c, "basic_player_status")
    ip, ifo = list_pages(c, "basic_player_info")
    if not sp:
        per_chunk[c] = {"status_pages": 0, "info_pages": len(ip)}
        log(c, "no status pages")
        continue
    assert len(sf) == 1, (c, sf)
    sk = keys_in_pages(sp)
    ik = keys_in_pages(ip) if ip else set()
    per_chunk[c] = {"window": sorted(sf)[0].split("basic_player_status.")[1], "status_pages": len(sp), "info_pages": len(ip),
                    "status_keys": len(sk), "info_keys": len(ik), "status_not_info": len(sk - ik), "info_not_status": len(ik - sk),
                    "status_keys_in_export": len(sk & keyset), "status_keys_not_in_export": len(sk - keyset)}
    rows += [(c, k, "status") for k in sk]
    rows += [(c, k, "info") for k in ik]
    log(c, json.dumps(per_chunk[c]))

mem = pl.DataFrame(rows, schema={"chunk": pl.String, "match_key": pl.String, "kind": pl.String}, orient="row")
mem.write_parquet(OUTD + "membership_long.parquet")
log("membership rows", mem.height)

# per export key: chunks holding it (status-based and info-based)
st_m = mem.filter(pl.col("kind") == "status").group_by("match_key").agg(pl.col("chunk").sort().alias("status_chunks"))
if_m = mem.filter(pl.col("kind") == "info").group_by("match_key").agg(pl.col("chunk").sort().alias("info_chunks"))
key_df = pl.DataFrame({"match_key": keys}).join(st_m, on="match_key", how="left").join(if_m, on="match_key", how="left")
key_df = key_df.with_columns(pl.col("status_chunks").list.len().fill_null(0).alias("n_status_chunks"),
                             pl.col("info_chunks").list.len().fill_null(0).alias("n_info_chunks"))
key_df = key_df.with_columns(pl.col("status_chunks").list.eval(pl.element()).alias("status_chunks"))
key_df = key_df.join(st_csv.select(["match_key", "split", "date", "build_status", "reason"]), on="match_key", how="left")
key_df.write_parquet(OUTD + "key_chunk_map.parquet")

summary = {"per_chunk": per_chunk, "elapsed_sec": round(time.time() - T0, 1)}
# group summary
groups = key_df.group_by("reason").agg([
    pl.len().alias("keys"),
    (pl.col("n_status_chunks") == 0).sum().alias("keys_in_no_chunk_status"),
    (pl.col("n_info_chunks") == 0).sum().alias("keys_in_no_chunk_info"),
    (pl.col("n_status_chunks") > 1).sum().alias("keys_in_more_than_1_chunk"),
    (pl.col("n_status_chunks") == 1).sum().alias("keys_in_exactly_1_chunk"),
    pl.col("n_status_chunks").max().alias("max_chunks_per_key"),
    pl.col("status_chunks").list.eval(pl.element().is_in(["c063", "c064", "c065", "c066", "c067"])).list.any().sum().alias("keys_with_any_c063_c067"),
]).sort("reason")
print("GROUP SUMMARY (by reason; keys counted once)")
print(groups)
summary["groups"] = groups.to_dicts()

# per chunk x group counts (keys present in that chunk, status-based)
exploded = key_df.select(["match_key", "reason", "status_chunks"]).explode("status_chunks").rename({"status_chunks": "chunk"}).filter(pl.col("chunk").is_not_null())
cg = exploded.group_by(["chunk", "reason"]).agg(pl.len().alias("keys")).pivot(on="reason", index="chunk", values="keys").fill_null(0).sort("chunk")
cg.write_csv(OUTD + "chunk_by_reason_counts.csv")
print("PER CHUNK x REASON (status-based key counts)")
print(cg)
summary["chunk_by_reason"] = cg.to_dicts()

# agreement of status vs info key sets (across all chunks)
disagree = key_df.filter(pl.col("n_status_chunks") != pl.col("n_info_chunks")).height
summary["status_vs_info_chunk_count_disagree_keys"] = disagree
summary["export_keys_in_no_chunk_status"] = key_df.filter(pl.col("n_status_chunks") == 0).height
json.dump(summary, open(OUTD + "summary.json", "w"), indent=1, default=str)
log("DONE; summary written")
