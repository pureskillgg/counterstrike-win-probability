"""Scope scan for the freeze-tick roster defect (read-only; S3 channel c000-c062; bucket matches of export_clean_v2).

For every export round of a bucket match: the players alive at the freeze tick (basic_player_status, health > 0 at tick == freeze)
and their roster rows (basic_player_info, all rows of the round). Defect flags per round:
  dup_alive  = alive pids whose info rows carry more than one team_code
  unro_alive = alive pids with no info row in the round
A round is affected when dup_alive or unro_alive is non-empty. Writes scope_rounds.parquet, scope_summary.json, affected_matches.csv.
"""
import io
import json
import time
import numpy as np
import polars as pl
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
FIN = "/home/sagemaker-user/adam/builder/final/"
OUT = "/home/sagemaker-user/adam/builder/fixes/freeze_scope/"
CHUNKS = ["c%03d" % i for i in range(0, 63)]
s3 = boto3.client("s3", config=Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "standard"}))
POOL = ThreadPoolExecutor(12)


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, flush=True)


keys = json.load(open(EXP + "match_keys.json"))
rm = np.load(EXP + "round_match.npy").astype(np.int64)
rn = np.load(EXP + "round_num.npy").astype(np.int64)
rf = np.load(EXP + "round_freeze.npy").astype(np.int64)
cr = pl.read_parquet(FIN + "clean_rounds.parquet", columns=["match_key", "use", "src"]).filter(pl.col("use") & (pl.col("src") == "bucket"))
bkeys = set(cr["match_key"].unique().to_list())
bm = np.array([keys[m] in bkeys for m in rm.tolist()])
rounds = pl.DataFrame({"match_key": [keys[m] for m in rm[bm].tolist()], "round": rn[bm], "freeze": rf[bm]},
                      schema={"match_key": pl.String, "round": pl.Int64, "freeze": pl.Int64})
wanted = sorted(rounds["match_key"].unique().to_list())
log("bucket export matches", len(wanted), "rounds", rounds.height)


def list_pages(c, kind):
    prefix = f"{PREFIX}/{c}/tome/csds/{kind}."
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            d, n = o["Key"].rsplit("/", 1)
            if d.rsplit("/", 1)[-1].startswith(kind + ".") and n.startswith("dataframe_"):
                out.append(o["Key"])
    return sorted(out)


def get_bytes(key):
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def read_pages(pkeys, columns):
    frames = []
    for b in POOL.map(get_bytes, pkeys):
        f = pl.read_parquet(io.BytesIO(b))
        frames.append(f.select(columns))
    return frames


INFO_COLS = ["match_key", "round", "player_id", "team_code"]
ST_COLS = ["match_key", "round", "tick", "player_id", "health"]
info_all, alive_all = [], []
chunk_stats = []
for c in CHUNKS:
    t1 = time.time()
    ik = list_pages(c, "basic_player_info")
    sk = list_pages(c, "basic_player_status")
    if not sk:
        continue
    info = pl.concat(read_pages(ik, INFO_COLS)).filter(pl.col("match_key").is_in(wanted)).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("team_code").cast(pl.Int64)).unique(
        subset=["match_key", "round", "player_id", "team_code"], keep="first")
    info_all.append(info)
    st = pl.concat(read_pages(sk, ST_COLS)).filter(pl.col("match_key").is_in(wanted)).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    st = st.join(rounds.select("match_key", "round", "freeze"), on=["match_key", "round"], how="inner").filter(
        (pl.col("tick") == pl.col("freeze")) & (pl.col("health") > 0)).select("match_key", "round", "player_id").unique()
    alive_all.append(st)
    chunk_stats.append({"chunk": c, "info_rows": info.height, "alive_rows": st.height, "sec": round(time.time() - t1, 1)})
    log(json.dumps(chunk_stats[-1]))

info = pl.concat(info_all).unique(subset=["match_key", "round", "player_id", "team_code"], keep="first")
alive = pl.concat(alive_all).unique()
codes = info.group_by(["match_key", "round", "player_id"]).agg(pl.col("team_code").n_unique().alias("n_codes"))
A = alive.join(codes, on=["match_key", "round", "player_id"], how="left").with_columns(pl.col("n_codes").fill_null(0))
dup = A.filter(pl.col("n_codes") > 1).group_by(["match_key", "round"]).agg(pl.len().alias("dup_alive"))
unro = A.filter(pl.col("n_codes") == 0).group_by(["match_key", "round"]).agg(pl.len().alias("unro_alive"))
n_alive = alive.group_by(["match_key", "round"]).agg(pl.len().alias("alive_n"))
R = (rounds.join(n_alive, on=["match_key", "round"], how="left").join(dup, on=["match_key", "round"], how="left")
     .join(unro, on=["match_key", "round"], how="left")
     .with_columns(pl.col("alive_n").fill_null(0), pl.col("dup_alive").fill_null(0), pl.col("unro_alive").fill_null(0))
     .with_columns(((pl.col("dup_alive") > 0) | (pl.col("unro_alive") > 0)).alias("affected")))
R.write_parquet(OUT + "scope_rounds.parquet")
aff = R.filter(pl.col("affected"))
summ = {"bucket_export_matches": len(wanted), "rounds": R.height,
        "rounds_with_alive_rows": int((R["alive_n"] > 0).sum()),
        "affected_rounds": aff.height, "affected_matches": aff["match_key"].n_unique(),
        "rounds_with_dup_alive": int((R["dup_alive"] > 0).sum()), "dup_alive_pairs": int(R["dup_alive"].sum()),
        "rounds_with_unrostered_alive": int((R["unro_alive"] > 0).sum()), "unrostered_alive_players": int(R["unro_alive"].sum()),
        "rounds_alive_not_10": int((R["alive_n"] != 10).sum()),
        "rounds_alive_not_10_and_affected": int(((R["alive_n"] != 10) & R["affected"]).sum()),
        "rounds_alive_not_10_unaffected": int(((R["alive_n"] != 10) & ~R["affected"]).sum()),
        "chunks": chunk_stats, "sec": round(time.time() - T0, 1)}
json.dump(summ, open(OUT + "scope_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "chunks"}, indent=1), flush=True)
aff.group_by("match_key").agg(pl.len().alias("rounds")).sort("rounds", descending=True).write_csv(OUT + "affected_matches.csv")
log("done")
