"""Position features for the bucket matches, every other second, computed with the laptop's own feature function.

Per match: raw player_vector (positions), player_status (alive, defuser), player_info (side), player_blind (flash events).
The wanted seconds are the export's seconds with an even index inside the round (k = 0, 2, 4, ...), at tick = freeze + 64 k.
Writes final/pos_parts/bucket_cNNN.parquet: match_key, round, tick, and the 24 float32 features (nulls for undefined).
  ~/venv-dsdk/bin/python pos_bucket.py c000 c001 ...
"""
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import polars as pl
from botocore.config import Config

sys.path.insert(0, os.path.expanduser("~/adam/builder/posport/scripts/data"))
import make_pos_features as MPF  # noqa: E402  (the laptop's feature code, unchanged)

B = "pureskillgg-csgo-production-sagemaker-0"
BUILDER = os.path.expanduser("~/adam/builder")
FIN = BUILDER + "/final"
EXP = FIN + "/export_clean"
SCR = "/mnt/sagemaker-nvme/posb"
OUTD = os.environ.get("POSB_OUT", FIN + "/pos_parts")  # POSB_OUT: write the chunks elsewhere (identity checks)
os.makedirs(OUTD, exist_ok=True)
cli = boto3.client("s3", config=Config(max_pool_connections=64, retries={"max_attempts": 8},
                                       request_checksum_calculation="when_required",
                                       response_checksum_validation="when_required"))
t0 = time.time()
keys = json.load(open(EXP + "/match_keys.json"))
kid = {k: i for i, k in enumerate(keys)}
off = np.load(EXP + "/offsets.npy")
rm = np.load(EXP + "/round_match.npy")
rn = np.load(EXP + "/round_num.npy")
frz = np.load(EXP + "/round_freeze.npy").astype(np.int64)
plant = np.load(EXP + "/round_plant.npy").astype(np.int64)
n_steps = np.diff(off)
order = np.argsort(rm, kind="stable")
bounds = np.concatenate([[0], np.cumsum(np.bincount(rm, minlength=len(keys)))])
sites = {m: np.array(v, dtype="float64") for m, v in json.load(open(FIN + "/sites_v1.json")).items()}
maps = json.load(open(EXP + "/maps.json"))
rmap = np.load(EXP + "/round_map.npy")
map_of = {}
for i in range(len(rm)):
    map_of.setdefault(int(rm[i]), maps[int(rmap[i])])
plants = pl.read_parquet(FIN + "/plants_bucket.parquet")
# one row per bomb_begin_plant event (tick, position): feats_for_day looks the bomb up per tick (latest plant at or before it)
pxy = (plants.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).sort("tick", maintain_order=True)
       .unique(["match_key", "round", "tick"], keep="last", maintain_order=True)
       .select("match_key", "round", "tick", pl.col("x").alias("bx"), pl.col("y").alias("by")))
cr = pl.read_parquet(FIN + "/clean_rounds.parquet", columns=["match_key", "use", "src"]).filter(pl.col("use") & (pl.col("src") == "bucket"))
bkeys = set(cr["match_key"].unique().to_list())


def fetch(folder, name):
    dst = f"{SCR}/{folder}/{name}"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cli.download_file(B, f"data/{folder}/{name}", dst)
    return dst


def one_match(mk):
    mid = kid[mk]
    idx = order[bounds[mid]:bounds[mid + 1]]
    folder = mk.rsplit("/", 1)[0]
    try:
        vp = fetch(folder, "player_vector")
        sp = fetch(folder, "player_status")
        ip = fetch(folder, "player_info")
        bp = fetch(folder, "player_blind") if cr.height else None
    except Exception as e:  # a missing channel is logged, the match is skipped
        return ("skip", mk, repr(e)[:120])
    rows = []
    for i in idx:
        f, p, n = int(frz[i]), int(plant[i]), int(n_steps[i])
        ks = np.arange(0, n, 2, dtype=np.int64)
        tick = f + 64 * ks
        rows.append(pl.DataFrame({"round": np.full(len(ks), int(rn[i]), dtype=np.int64), "tick": tick, "second": ks,
                                  "planted": ((p >= 0) & (tick >= p)).astype(np.int64),
                                  "since": np.where((p >= 0) & (tick >= p), (tick - p) / 64.0, 0.0)}))
    W = pl.concat(rows).with_columns(pl.lit(mk).alias("match_key"), pl.lit(map_of.get(mid, "")).alias("map_name"))
    W = W.with_columns(pl.col("planted").cast(pl.Int64).alias("bomb_planted"), pl.col("since").cast(pl.Float64).alias("seconds_since_plant"))
    v = pl.read_parquet(vp, columns=["tick", "round", "player_id", "x_pos", "y_pos", "theta_ang", "team_code"]) \
        .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    v = v.unique(["round", "tick", "player_id"], keep="first", maintain_order=True)
    st = pl.read_parquet(sp, columns=["tick", "round", "player_id", "health", "has_defuser", "is_defusing"]) \
        .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    st = st.unique(["round", "tick", "player_id"], keep="first", maintain_order=True)
    wk = W.select("round", "tick", "second", "match_key", "map_name", "bomb_planted", "seconds_since_plant")
    v = v.join(wk, on=["round", "tick"], how="inner")
    st = st.join(wk, on=["round", "tick"], how="inner").filter(pl.col("health") > 0)  # alive only, as the laptop's 1 Hz tables
    ro = pl.read_parquet(ip, columns=["round", "player_id", "team_code"]).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)).unique(["round", "player_id"], keep="last")
    pos = (v.join(st.select("round", "tick", "player_id", "health", "has_defuser", "is_defusing"), on=["round", "tick", "player_id"], how="inner")
           .join(ro.rename({"team_code": "roster_team"}), on=["round", "player_id"], how="left")
           .with_columns(pl.coalesce([pl.col("roster_team"), pl.col("team_code")]).cast(pl.Int64).alias("side"))
           .filter(pl.col("side").is_in([2, 3]))
           .rename({"x_pos": "x", "y_pos": "y", "theta_ang": "yaw"}))
    pos = pos.select("match_key", "round", "tick", "second", "player_id", "side", "x", "y", "yaw")
    sta = st.select("match_key", "round", "tick", "player_id", "health", "has_defuser", "is_defusing")
    ev = None
    if bp is not None:
        ev = pl.read_parquet(bp, columns=["round", "tick", "player_id", "blind_duration"]).with_columns(
            pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),
            pl.col("blind_duration").cast(pl.Float32), pl.lit(mk).alias("match_key")).select("match_key", "round", "tick", "player_id", "blind_duration")
    G = W.select("match_key", "round", "tick", "bomb_planted", "seconds_since_plant", "map_name")
    shutil.rmtree(f"{SCR}/{folder}", ignore_errors=True)
    return ("ok", pos, sta, G, ev)


def run_chunk(c):
    out_path = f"{OUTD}/bucket_{c}.parquet"
    if os.path.exists(out_path):
        return f"{c} exists"
    tc = time.time()
    mk_list = pl.read_parquet(f"{BUILDER}/data/{c}/matches.parquet", columns=["match_key"])["match_key"].to_list()
    todo = [m for m in mk_list if m in bkeys and m in kid]
    if not todo:
        return f"{c} no usable bucket matches"
    feats, skipped = [], []
    BLOCK = 60
    for b0 in range(0, len(todo), BLOCK):
        block = todo[b0:b0 + BLOCK]
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(one_match, block))
        ok = [r for r in res if r[0] == "ok"]
        skipped += [(r[1], r[2]) for r in res if r[0] == "skip"]
        if not ok:
            continue
        pos = pl.concat([r[1] for r in ok], how="vertical_relaxed")
        sta = pl.concat([r[2] for r in ok], how="vertical_relaxed")
        G = pl.concat([r[3] for r in ok], how="vertical_relaxed")
        evs = [r[4] for r in ok if r[4] is not None]
        ev = pl.concat(evs, how="vertical_relaxed") if evs else None
        bm = set(block)
        pxy_b = pxy.filter(pl.col("match_key").is_in(list(bm)))
        f = MPF.feats_for_day(pos, sta, G, pxy_b, sites, 2, ev)
        feats.append(f)
    if not feats:
        return f"{c} no rows"
    out = pl.concat(feats, how="vertical_relaxed")
    out.write_parquet(out_path)
    msg = f"{c} matches {len(todo)} skipped {len(skipped)} rows {out.height} secs {time.time() - tc:.0f}"
    if skipped:
        print("  skipped sample", skipped[:3], flush=True)
    return msg


for c in sys.argv[1:]:
    print(run_chunk(c), f"| total {time.time() - t0:.0f}s", flush=True)
