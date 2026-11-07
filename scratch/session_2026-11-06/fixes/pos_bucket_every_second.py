"""Position features for the bucket matches at EVERY second (k = 0, 1, 2, ...). Copy of pos_bucket.py, extended to the odd seconds.

Pinned to the code that built final/pos_parts (the even-second build):
  - feature module: make_pos_features.py.pre_causal (copied to fixes/pos_every_second/feat, the same file the even parts used);
  - bomb lookup: the last bomb_begin_plant of each round, as in pos_bucket.py.pre_causal.
  The causal per-tick bomb lookup in posport/ is not used here; it is a separate, pending change.

Per block of 60 matches, two passes over the same downloaded match files:
  even pass  k = 0, 2, 4, ...  same inputs and same every=2 call as pos_bucket.py, so these rows should match final/pos_parts exactly;
  odd pass   k = 1, 3, 5, ...  same feature function with every=1.
Output rows carry `second` (the export step index k; tick = freeze + 64 k).

  ~/venv-dsdk/bin/python pos_bucket_every_second.py c000 c001 ...
Writes <OUTD>/bucket_cNNN.parquet, OUTD = final/pos_all_seconds (POSB_OUT overrides it for tests).
Never writes export_clean_v2 or final/pos_parts.
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

FEAT = os.environ.get("POSB_FEAT", os.path.expanduser("~/adam/builder/fixes/pos_every_second/feat"))
sys.path.insert(0, FEAT)
import make_pos_features as MPF  # noqa: E402  (pinned copy of the laptop feature code, pre-causal bomb lookup)

B = "pureskillgg-csgo-production-sagemaker-0"
BUILDER = os.path.expanduser("~/adam/builder")
FIN = BUILDER + "/final"
EXP = FIN + "/export_clean"  # the grid the existing pos_parts were built on
SCR = os.environ.get("POSB_SCR", "/mnt/sagemaker-nvme/posb_all_seconds")
OUTD = os.environ.get("POSB_OUT", FIN + "/pos_all_seconds")
if "export_clean_v2" in OUTD or "export_clean_v3" in OUTD or os.path.abspath(OUTD) == os.path.abspath(FIN + "/pos_parts"):
    sys.exit("refusing to write to " + OUTD)
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
# pre-causal lookup (pos_bucket.py.pre_causal): the last bomb_begin_plant of each round, one row per round
pxy = (plants.sort("tick").group_by("match_key", "round", maintain_order=True)
       .agg(pl.col("x").last().alias("bx"), pl.col("y").last().alias("by")).with_columns(pl.col("round").cast(pl.Int64)))
cr = pl.read_parquet(FIN + "/clean_rounds.parquet", columns=["match_key", "use", "src"]).filter(pl.col("use") & (pl.col("src") == "bucket"))
bkeys = set(cr["match_key"].unique().to_list())


def fetch(folder, name):
    dst = f"{SCR}/{folder}/{name}"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cli.download_file(B, f"data/{folder}/{name}", dst)
    return dst


def steps_frame(mk, mid, idx, par):
    """par 0: even seconds (the pos_bucket.py grid); par 1: odd seconds. Same construction for both."""
    rows = []
    for i in idx:
        f, p, n = int(frz[i]), int(plant[i]), int(n_steps[i])
        ks = np.arange(par, n, 2, dtype=np.int64)
        tick = f + 64 * ks
        rows.append(pl.DataFrame({"round": np.full(len(ks), int(rn[i]), dtype=np.int64), "tick": tick, "second": ks,
                                  "planted": ((p >= 0) & (tick >= p)).astype(np.int64),
                                  "since": np.where((p >= 0) & (tick >= p), (tick - p) / 64.0, 0.0)}))
    W = pl.concat(rows).with_columns(pl.lit(mk).alias("match_key"), pl.lit(map_of.get(mid, "")).alias("map_name"))
    return W.with_columns(pl.col("planted").cast(pl.Int64).alias("bomb_planted"), pl.col("since").cast(pl.Float64).alias("seconds_since_plant"))


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
    v0 = pl.read_parquet(vp, columns=["tick", "round", "player_id", "x_pos", "y_pos", "theta_ang", "team_code"]) \
        .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    v0 = v0.unique(["round", "tick", "player_id"], keep="first", maintain_order=True)
    st0 = pl.read_parquet(sp, columns=["tick", "round", "player_id", "health", "has_defuser", "is_defusing"]) \
        .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    st0 = st0.unique(["round", "tick", "player_id"], keep="first", maintain_order=True)
    ro = pl.read_parquet(ip, columns=["round", "player_id", "team_code"]).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)).unique(["round", "player_id"], keep="last")
    ev = None
    if bp is not None:
        ev = pl.read_parquet(bp, columns=["round", "tick", "player_id", "blind_duration"]).with_columns(
            pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),
            pl.col("blind_duration").cast(pl.Float32), pl.lit(mk).alias("match_key")).select("match_key", "round", "tick", "player_id", "blind_duration")
    parts = {}
    for par in (0, 1):
        W = steps_frame(mk, mid, idx, par)
        wk = W.select("round", "tick", "second", "match_key", "map_name", "bomb_planted", "seconds_since_plant")
        v = v0.join(wk, on=["round", "tick"], how="inner")
        st = st0.join(wk, on=["round", "tick"], how="inner").filter(pl.col("health") > 0)  # alive only, as the laptop's 1 Hz tables
        pos = (v.join(st.select("round", "tick", "player_id", "health", "has_defuser", "is_defusing"), on=["round", "tick", "player_id"], how="inner")
               .join(ro.rename({"team_code": "roster_team"}), on=["round", "player_id"], how="left")
               .with_columns(pl.coalesce([pl.col("roster_team"), pl.col("team_code")]).cast(pl.Int64).alias("side"))
               .filter(pl.col("side").is_in([2, 3]))
               .rename({"x_pos": "x", "y_pos": "y", "theta_ang": "yaw"}))
        pos = pos.select("match_key", "round", "tick", "second", "player_id", "side", "x", "y", "yaw")
        sta = st.select("match_key", "round", "tick", "player_id", "health", "has_defuser", "is_defusing")
        G = W.select("match_key", "round", "tick", "bomb_planted", "seconds_since_plant", "map_name")
        sec = W.select("match_key", "round", "tick", "second")
        parts[par] = (pos, sta, G, sec)
    shutil.rmtree(f"{SCR}/{folder}", ignore_errors=True)
    return ("ok", parts, ev)


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
        evs = [r[2] for r in ok if r[2] is not None]
        ev = pl.concat(evs, how="vertical_relaxed") if evs else None
        bm = set(block)
        pxy_b = pxy.filter(pl.col("match_key").is_in(list(bm)))
        for par, every in ((0, 2), (1, 1)):
            pos = pl.concat([r[1][par][0] for r in ok], how="vertical_relaxed")
            if pos.height == 0:
                continue
            sta = pl.concat([r[1][par][1] for r in ok], how="vertical_relaxed")
            G = pl.concat([r[1][par][2] for r in ok], how="vertical_relaxed")
            sec = pl.concat([r[1][par][3] for r in ok], how="vertical_relaxed").unique(["match_key", "round", "tick"])
            f = MPF.feats_for_day(pos, sta, G, pxy_b, sites, every, ev)
            f = f.join(sec, on=["match_key", "round", "tick"], how="left")
            if f["second"].null_count():
                raise RuntimeError(f"{c}: {f['second'].null_count()} rows without a second index")
            feats.append(f)
    if not feats:
        return f"{c} no rows"
    out = pl.concat(feats, how="vertical_relaxed").with_columns(pl.col("second").cast(pl.Int64))
    tmp = out_path + ".tmp"
    out.write_parquet(tmp)
    os.replace(tmp, out_path)
    n_even = out.filter(pl.col("second") % 2 == 0).height
    msg = (f"{c} matches {len(todo)} skipped {len(skipped)} rows {out.height} even {n_even} odd {out.height - n_even} "
           f"secs {time.time() - tc:.0f}")
    if skipped:
        print("  skipped sample", skipped[:3], flush=True)
    return msg


for c in sys.argv[1:]:
    print(run_chunk(c), f"| total {time.time() - t0:.0f}s", flush=True)
print("POS_ALL_SECONDS_DONE", flush=True)
