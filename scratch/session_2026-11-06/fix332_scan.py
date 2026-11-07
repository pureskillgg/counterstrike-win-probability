"""fix332: recount freeze-tick roster defects over the whole export (read-only on every export folder).

Raw source: S3 tome chunks c000-c062 (basic_player_info, basic_player_status), kept only for export rounds.
Per export round (round_match, round_num, round_freeze of export_clean_v4):
  alive        = status row at tick == freeze with health > 0 (player_id, unique)
  roster       = info rows of the round, unique (player_id, team_code)
  dup_roster   = player_id with > 1 distinct team_code in the round (definition 1, any alive state)
  unro_alive   = alive player_id with no info row in the round (definition 2)
  dup_alive    = alive player_id with > 1 distinct team_code (comparison with the earlier scope scan)
  alive_ct/t   = alive player_ids whose only roster code is 3 (CT) / 2 (T)
kind: dup (definition 1 only), unro (definition 2 only), dup+unro (both).
Writes only under fixes/fix332/.
"""
import io
import csv
import json
import os
import time

import numpy as np
import polars as pl
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
FIN = "/home/sagemaker-user/adam/builder/final/"
EXP = FIN + "export_clean_v4/"
FIX = "/home/sagemaker-user/adam/builder/fixes/"
OUT = FIX + "fix332/"
os.makedirs(OUT, exist_ok=True)
CHUNKS = ["c%03d" % i for i in range(63)]
s3 = boto3.client("s3", config=Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "standard"}))
POOL = ThreadPoolExecutor(12)


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, flush=True)


# ---------- export round table (read only) ----------
keys = json.load(open(EXP + "match_keys.json"))
rm = np.load(EXP + "round_match.npy").astype(np.int64)
rn = np.load(EXP + "round_num.npy").astype(np.int64)
rf = np.load(EXP + "round_freeze.npy").astype(np.int64)
N = len(rm)
KEYIDX = pl.DataFrame({"match_key": keys, "mid": np.arange(len(keys), dtype=np.int64)})
ER = pl.DataFrame({"ridx": np.arange(N, dtype=np.int64), "mid": rm, "round": rn, "freeze": rf})
ER = ER.join(KEYIDX, on="mid", how="left")
cm = pl.read_parquet(FIN + "clean_matches.parquet", columns=["match_key", "src"]).unique(subset=["match_key"], keep="first")
ER = ER.join(cm, on="match_key", how="left").with_columns(pl.col("src").fill_null("unknown"))
log("export rounds", ER.height, "matches", len(keys), "src", ER.group_by("src").len().rows())
ndup_rows = ER.group_by(["mid", "round"]).len().filter(pl.col("len") > 1).height
log("export (match, round) pairs appearing more than once:", ndup_rows)
ER_KEY = ER.select(["mid", "round", "freeze"]).unique(subset=["mid", "round"], keep="first")
freeze_neg = ER.filter(pl.col("freeze") <= 0).height
log("export rounds with freeze <= 0:", freeze_neg)


# ---------- S3 tome pages ----------
def list_pages(c, kind):
    prefix = f"{PREFIX}/{c}/tome/csds/{kind}."
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            d, n = o["Key"].rsplit("/", 1)
            if d.rsplit("/", 1)[-1].startswith(kind + ".") and n.startswith("dataframe_"):
                out.append(o["Key"])
    return sorted(out)


def get_bytes(k):
    return s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()


INFO_COLS = ["match_key", "round", "player_id", "team_code", "player_id_fixed"]
ST_COLS = ["match_key", "round", "tick", "player_id", "health"]
info_parts, freeze_parts = [], []
info_mids, stat_mids = set(), set()
chunk_stats = []
for c in CHUNKS:
    t1 = time.time()
    ik = list_pages(c, "basic_player_info")
    sk = list_pages(c, "basic_player_status")
    if not ik and not sk:
        continue
    n_info = n_freeze = 0
    for b in POOL.map(get_bytes, ik):
        f = pl.read_parquet(io.BytesIO(b), columns=INFO_COLS).join(KEYIDX, on="match_key", how="inner")
        if f.height == 0:
            continue
        info_mids.update(f["mid"].unique().to_list())
        f = f.with_columns(pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),
                           pl.col("team_code").cast(pl.Int64), pl.col("player_id_fixed").cast(pl.Int64))
        f = f.select(["mid", "round", "player_id", "team_code", "player_id_fixed"]).join(
            ER_KEY.select(["mid", "round"]), on=["mid", "round"], how="inner")
        n_info += f.height
        info_parts.append(f)
    for b in POOL.map(get_bytes, sk):
        f = pl.read_parquet(io.BytesIO(b), columns=ST_COLS).join(KEYIDX, on="match_key", how="inner")
        if f.height == 0:
            continue
        stat_mids.update(f["mid"].unique().to_list())
        f = f.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                           pl.col("player_id").cast(pl.Int64), pl.col("health").cast(pl.Int64))
        f = f.select(["mid", "round", "tick", "player_id", "health"]).join(ER_KEY, on=["mid", "round"], how="inner")
        f = f.filter(pl.col("tick") == pl.col("freeze")).select(["mid", "round", "player_id", "health"])
        if f.height:
            n_freeze += f.height
            freeze_parts.append(f)
    chunk_stats.append({"chunk": c, "info_rows_export": n_info, "freeze_rows": n_freeze, "sec": round(time.time() - t1, 1)})
    log(json.dumps(chunk_stats[-1]))

log("info parts", len(info_parts), "freeze parts", len(freeze_parts), "matches with info", len(info_mids), "matches with status", len(stat_mids))

# ---------- per round aggregation ----------
info = pl.concat(info_parts).unique(subset=["mid", "round", "player_id", "team_code"], keep="first")
log("roster pairs", info.height)
codes = info.group_by(["mid", "round", "player_id"]).agg(
    pl.col("team_code").n_unique().alias("n_codes"),
    pl.col("team_code").min().alias("code_min"),
    pl.col("team_code").max().alias("code_max"),
    (pl.col("player_id_fixed") != pl.col("player_id")).any().alias("pidfix_diff"),
)
fz = pl.concat(freeze_parts)
statpid = fz.unique(subset=["mid", "round", "player_id"], keep="first").group_by(["mid", "round"]).agg(
    pl.len().alias("n_status_pids"))
alive = fz.filter(pl.col("health") > 0).select(["mid", "round", "player_id"]).unique()
log("alive rows", alive.height, "status pids", statpid.height)

A = alive.join(codes.select(["mid", "round", "player_id", "n_codes", "code_min"]),
               on=["mid", "round", "player_id"], how="left").with_columns(
    pl.col("n_codes").fill_null(0), pl.col("code_min").fill_null(0))
A = A.with_columns(
    (pl.col("n_codes") == 0).alias("unro"),
    (pl.col("n_codes") > 1).alias("dup"),
    ((pl.col("n_codes") == 1) & (pl.col("code_min") == 3)).alias("sct"),
    ((pl.col("n_codes") == 1) & (pl.col("code_min") == 2)).alias("st"),
)
AR = A.group_by(["mid", "round"]).agg(
    pl.len().alias("alive_n"),
    pl.col("unro").sum().alias("unro_n"),
    pl.col("dup").sum().alias("dup_alive_n"),
    pl.col("sct").sum().alias("alive_ct"),
    pl.col("st").sum().alias("alive_t"),
)
DR = codes.filter(pl.col("n_codes") > 1)
DRN = DR.group_by(["mid", "round"]).agg(pl.len().alias("dup_roster_n"))
RN = info.group_by(["mid", "round"]).agg(pl.col("player_id").n_unique().alias("n_info_pids"))
PFD = codes.filter(pl.col("pidfix_diff")).group_by(["mid", "round"]).agg(pl.len().alias("pidfix_n"))

R = ER.join(RN, on=["mid", "round"], how="left").join(statpid, on=["mid", "round"], how="left").join(
    AR, on=["mid", "round"], how="left").join(DRN, on=["mid", "round"], how="left").join(
    PFD, on=["mid", "round"], how="left")
fills = ["n_info_pids", "n_status_pids", "alive_n", "unro_n", "dup_alive_n", "alive_ct", "alive_t", "dup_roster_n", "pidfix_n"]
R = R.with_columns([pl.col(c).fill_null(0) for c in fills])
R = R.with_columns(
    pl.when((pl.col("dup_roster_n") > 0) & (pl.col("unro_n") > 0)).then(pl.lit("dup+unro"))
    .when(pl.col("dup_roster_n") > 0).then(pl.lit("dup"))
    .when(pl.col("unro_n") > 0).then(pl.lit("unro"))
    .otherwise(pl.lit("")).alias("kind"),
    (pl.col("n_info_pids") > 0).alias("has_roster"),
    (pl.col("n_status_pids") > 0).alias("has_freeze_status"),
)
log("export rounds after join", R.height)
R.write_parquet(OUT + "round_flags.parquet")

# id lists for candidate rounds
dup_ids = {}
for mid, rd, pid in DR.select(["mid", "round", "player_id"]).rows():
    dup_ids.setdefault((mid, rd), []).append(pid)
dup_alive_ids = {}
for mid, rd, pid in A.filter(pl.col("dup")).select(["mid", "round", "player_id"]).rows():
    dup_alive_ids.setdefault((mid, rd), []).append(pid)
unro_ids = {}
for mid, rd, pid in A.filter(pl.col("unro")).select(["mid", "round", "player_id"]).rows():
    unro_ids.setdefault((mid, rd), []).append(pid)

cand = R.filter(pl.col("kind") != "").sort(["match_key", "round"])
log("candidate rounds", cand.height)
rows_main, rows_detail = [], []
seen = set()
dupe_candidate_rows = 0
for r in cand.select(["mid", "round", "freeze", "match_key", "src", "kind", "alive_n", "alive_ct", "alive_t",
                      "n_info_pids", "n_status_pids", "dup_alive_n", "unro_n", "dup_roster_n", "pidfix_n", "ridx"]).iter_rows(named=True):
    k = (r["mid"], r["round"])
    if k in seen:
        dupe_candidate_rows += 1
        continue
    seen.add(k)
    di = sorted(dup_ids.get(k, []))
    ui = sorted(unro_ids.get(k, []))
    da = sorted(dup_alive_ids.get(k, []))
    rows_main.append({"match_key": r["match_key"], "round": r["round"], "freeze_tick": r["freeze"], "kind": r["kind"],
                      "dup_ids": ";".join(str(x) for x in di), "unro_ids": ";".join(str(x) for x in ui),
                      "alive_ct": r["alive_ct"], "alive_t": r["alive_t"]})
    rows_detail.append({"match_key": r["match_key"], "round": r["round"], "freeze_tick": r["freeze"], "kind": r["kind"],
                        "src": r["src"], "ridx": r["ridx"], "dup_ids": ";".join(str(x) for x in di),
                        "dup_alive_ids": ";".join(str(x) for x in da), "unro_ids": ";".join(str(x) for x in ui),
                        "alive_ct": r["alive_ct"], "alive_t": r["alive_t"], "alive_n": r["alive_n"],
                        "n_info_pids": r["n_info_pids"], "n_status_pids": r["n_status_pids"],
                        "pidfix_n": r["pidfix_n"], "dup_alive_n": r["dup_alive_n"]})
log("duplicate candidate rows dropped", dupe_candidate_rows)

with open(OUT + "candidates.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["match_key", "round", "freeze_tick", "kind", "dup_ids", "unro_ids", "alive_ct", "alive_t"])
    w.writeheader()
    w.writerows(rows_main)
with open(OUT + "candidates_detail.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_detail[0].keys()))
    w.writeheader()
    w.writerows(rows_detail)
log("wrote candidates", len(rows_main))

# missing matches (no info rows or no freeze status in tome)
mm = ER.select(["mid", "match_key", "src"]).unique(subset=["mid"], keep="first").with_columns(
    pl.col("mid").is_in(list(info_mids)).alias("in_info"), pl.col("mid").is_in(list(stat_mids)).alias("in_status"))
miss = mm.filter(~pl.col("in_info") | ~pl.col("in_status"))
miss_rounds = ER.group_by("mid").len().rename({"len": "export_rounds"})
miss = miss.join(miss_rounds, on="mid", how="left")
miss.select(["match_key", "src", "in_info", "in_status", "export_rounds"]).write_csv(OUT + "missing_matches.csv")
log("export matches with no info rows in tome", mm.filter(~pl.col("in_info")).height,
    "| no status rows in tome", mm.filter(~pl.col("in_status")).height, "| by src", miss.group_by("src").len().rows())

# ---------- summary ----------
def cnt(expr, df=R):
    return df.filter(expr).height


summ = {
    "export_rounds": R.height, "export_matches": len(keys),
    "src_rounds": R.group_by("src").len().rows(),
    "export_matches_with_info_in_tome": len(info_mids), "export_matches_with_status_in_tome": len(stat_mids),
    "rounds_with_roster_rows": cnt(pl.col("has_roster")),
    "rounds_with_freeze_status": cnt(pl.col("has_freeze_status")),
    "rounds_no_freeze_status": cnt(~pl.col("has_freeze_status")),
    "rounds_no_roster": cnt(~pl.col("has_roster")),
    "def1_dup_roster_rounds": cnt(pl.col("dup_roster_n") > 0),
    "def1_dup_roster_pids": int(R["dup_roster_n"].sum()),
    "def2_unro_alive_rounds": cnt(pl.col("unro_n") > 0),
    "def2_unro_alive_players": int(R["unro_n"].sum()),
    "dup_alive_rounds_alivebased": cnt(pl.col("dup_alive_n") > 0),
    "alive_scope_rounds_dupalive_or_unro": cnt((pl.col("dup_alive_n") > 0) | (pl.col("unro_n") > 0)),
    "union_candidates_def1_or_def2": cnt(pl.col("kind") != ""),
    "kind_counts": R.filter(pl.col("kind") != "").group_by("kind").len().rows(),
    "kind_by_src": R.filter(pl.col("kind") != "").group_by(["src", "kind"]).len().rows(),
    "pidfix_diff_candidate_rounds": cnt((pl.col("kind") != "") & (pl.col("pidfix_n") > 0)),
    "candidate_rows_written": len(rows_main), "duplicate_candidate_rows_dropped": dupe_candidate_rows,
    "export_round_pairs_duplicated": ndup_rows, "freeze_le_zero_rounds": freeze_neg,
    "chunk_seconds": round(time.time() - T0, 1),
}
json.dump(summ, open(OUT + "summary.json", "w"), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str), flush=True)
log("DONE")
