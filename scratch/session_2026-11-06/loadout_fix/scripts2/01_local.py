import json, sys, time, glob, os
import numpy as np
import polars as pl
sys.path.insert(0, "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix/scripts2")
from loadout_rule import aggregate, side_index, FIELDS, INPUT_COLS, TICK

S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
OUT = S + "/out"
T0 = time.time()


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, flush=True)


keys = json.load(open(S + "/match_keys.json"))
split = np.load(S + "/export/match_split.npy")
off = np.load(S + "/offsets.npy").astype(np.int64)
rm = np.load(S + "/round_match.npy").astype(np.int64)
rn = np.load(S + "/round_num.npy").astype(np.int64)
rf = np.load(S + "/round_freeze.npy").astype(np.int64)
nsteps = np.diff(off)
dates = [k.split("/")[1] + "-" + k.split("/")[2] + "-" + k.split("/")[3] for k in keys]
# scope: train (split code 0) and validation (code 1) only. Test (code 2) is coverage-only, never built.
# Path date from 2026-10-21 is protected (holdout B match dates Oct 21-25; Oct 26-31 holdout).
PROTECT_FROM = "2026-10-21"
in_scope = np.array([(split[i] in (0, 1)) and (dates[i] < PROTECT_FROM) for i in range(len(keys))])
log("in-scope matches", int(in_scope.sum()), "of", len(keys))
mt = pl.DataFrame({"match_key": keys, "m": np.arange(len(keys), dtype=np.int64), "split": split.astype(np.int64),
                   "date": dates, "in_scope": in_scope})
mt_s = mt.filter(pl.col("in_scope"))
rt = pl.DataFrame({"m": rm, "round": rn, "r": np.arange(len(rm), dtype=np.int64), "freeze": rf,
                   "off_r": off[:-1], "nsteps": nsteps})
rt_s = rt.join(mt_s.select("m"), on="m", how="inner")
log("in-scope rounds", rt_s.height)

NS = int(off[-1])
arr = np.lib.format.open_memmap(OUT + "/loadout_step_i16.npy", mode="w+", dtype=np.int16, shape=(NS, 2, len(FIELDS) - 1))
arr_eq = np.lib.format.open_memmap(OUT + "/loadout_equip_i32.npy", mode="w+", dtype=np.int32, shape=(NS, 2))
log("memmap created", arr.shape, arr_eq.shape)

files = []
for f in sorted(glob.glob("G:/pureskill/csds-cs2/data/status_1hz/2026-*.parquet")):
    name = os.path.basename(f)[:10]
    if "2026-09-01" <= name <= "2026-10-20":
        files.append(("local_1hz", f))
for f in sorted(glob.glob("G:/pureskill/csds-cs2/data/status_1hz_2024/*.parquet")):
    files.append(("local_1hz_2024", f))
log("files", len(files))

cov_parts = []
file_stats = []
seen_src = {}
for src, f in files:
    t1 = time.time()
    raw = pl.read_parquet(f, columns=["match_key", "round", "tick", "player_id", "side"] + INPUT_COLS)
    n_raw = raw.height
    d = raw.join(mt_s.select("match_key", "m"), on="match_key", how="inner")
    del raw
    n_match = d.height
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
    cov = cov.with_columns(pl.lit(src).alias("src"), pl.lit(os.path.basename(f)).alias("file"))
    cov_parts.append(cov)
    for mm in cov["m"].to_list():
        if mm in seen_src and seen_src[mm] != src:
            log("WARNING match in two sources", mm, seen_src[mm], src)
        seen_src[mm] = src
    agg = aggregate(d).sort(["s", "side_idx"])
    s_idx = agg["s"].to_numpy()
    side_idx = agg["side_idx"].to_numpy()
    vals = np.stack([agg[c].to_numpy() for c in FIELDS], axis=1)
    v16 = vals[:, :-1]
    eq = vals[:, -1]
    if v16.size and (v16.max() > 32767 or v16.min() < -32768):
        raise SystemExit("int16 overflow in " + f)
    if eq.size and (eq.max() > 2147483647 or eq.min() < -2147483648):
        raise SystemExit("int32 overflow in " + f)
    arr[s_idx, side_idx, :] = v16.astype(np.int16)
    arr_eq[s_idx, side_idx] = eq.astype(np.int32)
    file_eq_max = int(eq.max()) if eq.size else 0
    st = {"src": src, "file": os.path.basename(f), "raw_rows": n_raw, "rows_in_scope_matches": n_match,
          "rows_no_export_round": n_no_round, "rows_pre_freeze": n_pre_freeze, "rows_off_grid": n_off_grid,
          "rows_beyond_round_steps": n_beyond, "rows_bad_side": n_bad_side, "rows_kept_before_dedupe": n_pre_dedupe,
          "duplicates_dropped": n_dups, "agg_rows_written": int(len(s_idx)), "matches": int(cov.height),
          "equip_max_step_side": file_eq_max, "sec": round(time.time() - t1, 1)}
    file_stats.append(st)
    log(json.dumps(st))
    del d, agg, cov
arr.flush()
arr_eq.flush()
cov_all = pl.concat(cov_parts)
cov_all = cov_all.join(mt.select("m", "match_key", "split", "date"), on="m", how="left")
cov_all.write_parquet(OUT + "/coverage_local.parquet")
json.dump(file_stats, open(OUT + "/local_file_stats.json", "w"), indent=1)
log("done; coverage rows", cov_all.height, "distinct matches", cov_all["m"].n_unique())
