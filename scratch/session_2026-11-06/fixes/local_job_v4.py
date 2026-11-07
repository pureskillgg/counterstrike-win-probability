# v4 local 1 Hz stage: the 1,164 test-split keys that no S3 chunk holds. Same rule as v3 01_space_body.py (loadout_rule.py).
# Reads only the 1 Hz files dated 2026-09-01 .. 2026-10-25 (never Oct 26-31), and only the rows of the 1,164 keys.
import json, sys, time, glob, os
import numpy as np
import polars as pl

V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
V4S = V4 + "scripts/"
sys.path.insert(0, V4S)
from loadout_rule import TICK, FIELDS, INPUT_COLS, aggregate, side_index

T0 = time.time()


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, file=sys.stderr, flush=True)


EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
SRC = "/home/sagemaker-user/adam/builder/laptop_status/status_1hz/"
LAST_DATE = "2026-10-25"

keys = json.load(open(EXP + "match_keys.json"))
off = np.load(EXP + "offsets.npy").astype(np.int64)
rm = np.load(EXP + "round_match.npy").astype(np.int64)
rn = np.load(EXP + "round_num.npy").astype(np.int64)
rf = np.load(EXP + "round_freeze.npy").astype(np.int64)
nsteps = np.diff(off)
NS = int(off[-1])
dates = [k.split("/")[1] + "-" + k.split("/")[2] + "-" + k.split("/")[3] for k in keys]
LOCAL_REQ = json.load(open(V4S + "local_target_v4.json"))
assert len(LOCAL_REQ) == 1164 and len(set(LOCAL_REQ)) == 1164
req = set(LOCAL_REQ)
keyset = set(keys)
assert req <= keyset
req_dates = [dates[i] for i, k in enumerate(keys) if k in req]
mn, mx = min(req_dates), max(req_dates)
log("local targets", len(LOCAL_REQ), "date range", mn, mx)
assert mx <= LAST_DATE

mt = pl.DataFrame({"match_key": keys, "m": np.arange(len(keys), dtype=np.int64)}).filter(pl.col("match_key").is_in(LOCAL_REQ))
assert mt.height == 1164
rt = pl.DataFrame({"m": rm, "round": rn, "r": np.arange(len(rm), dtype=np.int64), "freeze": rf,
                   "off_r": off[:-1], "nsteps": nsteps})
rt_s = rt.join(mt.select("m"), on="m", how="inner")
log("local-target rounds", rt_s.height)

arr = np.load(V4 + "loadout_step_i16.npy", mmap_mode="r+")
arr_eq = np.load(V4 + "loadout_equip_i32.npy", mmap_mode="r+")
assert arr.shape == (NS, 2, len(FIELDS) - 1) and arr_eq.shape == (NS, 2), (arr.shape, arr_eq.shape)

files = []
for f in sorted(glob.glob(SRC + "2026-*.parquet")):
    name = os.path.basename(f)[:10]
    if mn <= name <= LAST_DATE:
        files.append(f)
log("files to read", len(files), os.path.basename(files[0]) if files else None, os.path.basename(files[-1]) if files else None)

cov_parts = []
file_stats = []
file_of_m = {}
for f in files:
    t1 = time.time()
    lf = pl.scan_parquet(f)
    n_total = lf.select(pl.len()).collect().item()
    raw = lf.filter(pl.col("match_key").is_in(LOCAL_REQ)).select(
        ["match_key", "round", "tick", "player_id", "side"] + INPUT_COLS).collect()
    n_raw = raw.height
    d = raw.join(mt.select("match_key", "m"), on="match_key", how="inner")
    del raw
    d = d.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    d = d.join(rt_s.select("m", "round", "r", "freeze", "off_r", "nsteps"), on=["m", "round"], how="left")
    n_no_round = d.filter(pl.col("r").is_null()).height
    d = d.filter(pl.col("r").is_not_null())
    d = d.with_columns((pl.col("tick") - pl.col("freeze")).alias("dd"))
    n_pre_freeze = d.filter(pl.col("dd") < 0).height
    n_off_grid = d.filter((pl.col("dd") >= 0) & (pl.col("dd") % TICK != 0)).height
    d = d.filter((pl.col("dd") >= 0) & (pl.col("dd") % TICK == 0))
    d = d.with_columns((pl.col("dd") // TICK).alias("k"))
    n_beyond = d.filter(pl.col("k") >= pl.col("nsteps")).height
    d = d.filter(pl.col("k") < pl.col("nsteps"))
    d = d.with_columns((pl.col("off_r") + pl.col("k")).alias("s"), side_index(pl.col("side")).alias("side_idx"))
    n_bad_side = d.filter(pl.col("side_idx") < 0).height
    d = d.filter(pl.col("side_idx") >= 0)
    n_pre_dedupe = d.height
    d = d.unique(subset=["r", "tick", "player_id"], keep="first", maintain_order=False)
    n_dups = n_pre_dedupe - d.height
    cov = d.group_by("m").agg([pl.len().alias("rows"), pl.col("r").n_unique().alias("rounds_with_rows"),
                               pl.col("s").n_unique().alias("steps_with_rows"),
                               (pl.col("k") == 0).sum().alias("freeze_rows")])
    cov = cov.with_columns(pl.lit(os.path.basename(f)).alias("file"))
    for mm in cov["m"].to_list():
        if mm in file_of_m and file_of_m[mm] != os.path.basename(f):
            log("WARNING match in two files", mm, file_of_m[mm], os.path.basename(f))
        file_of_m[mm] = os.path.basename(f)
    cov_parts.append(cov)
    agg = aggregate(d).sort(["s", "side_idx"])
    s_idx = agg["s"].to_numpy()
    side_idx = agg["side_idx"].to_numpy()
    vals = np.stack([agg[cn].to_numpy() for cn in FIELDS], axis=1)
    v16 = vals[:, :-1]
    eq = vals[:, -1]
    if v16.size:
        assert v16.max() <= 32767 and v16.min() >= -32768 and eq.max() <= 2147483647
    arr[s_idx, side_idx, :] = v16.astype(np.int16)
    arr_eq[s_idx, side_idx] = eq.astype(np.int32)
    st = {"file": os.path.basename(f), "rows_in_file_metadata": n_total, "rows_req_matches": n_raw,
          "rows_no_export_round": n_no_round, "rows_pre_freeze": n_pre_freeze, "rows_off_grid": n_off_grid,
          "rows_beyond_round_steps": n_beyond, "rows_bad_side": n_bad_side, "rows_kept_before_dedupe": n_pre_dedupe,
          "duplicates_dropped": n_dups, "agg_rows_written": int(len(s_idx)), "matches": int(cov.height),
          "sec": round(time.time() - t1, 1)}
    file_stats.append(st)
    log(json.dumps(st))
    del d, agg, cov
arr.flush()
arr_eq.flush()
cov_all = pl.concat(cov_parts).join(mt.select("m", "match_key"), on="m", how="left")
cov_all.write_parquet(V4S + "coverage_local_v4.parquet")
json.dump(file_stats, open(V4S + "local_file_stats_v4.json", "w"), indent=1)
log("LOCAL STAGE DONE; coverage rows", cov_all.height, "distinct matches", cov_all["m"].n_unique(), "of", len(LOCAL_REQ))
