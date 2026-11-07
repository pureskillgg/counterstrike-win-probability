"""Single rule set for the per-second loadout (serving rule from scripts/serving/win_probability.py, lines 40-57 and 249-291).

Grid: tick = freeze + 64*k for k >= 0 (freeze = export round_freeze), k < steps of the round in the export (serving loop bound).
Per (step, side): the 20 fields below, computed exactly as serving does (null -> 0, inv_primary null -> 0, false for null flags).
Side: team_code 3 = CT, 2 = T (serving: any code other than 3 goes to T; here any other code is dropped and counted).
"""
import polars as pl

TICK = 64
RIFLE = (7, 8, 10, 13, 16, 39, 60)
M4 = (16, 60)
AK = (7,)
AWP = (9,)
SNIPER_OTHER = (11, 38, 40)
SMG = (17, 19, 23, 24, 26, 33, 34)
HEAVY = (14, 25, 27, 28, 29, 35)
FIELDS = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
          "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy", "equip"]
INPUT_COLS = ["health", "armor", "has_helmet", "has_defuser", "is_defusing", "inv_primary", "inv_flashbang",
              "inv_smokegrenade", "inv_hegrenade", "inv_molotov", "inv_incgrenade", "inv_decoy", "equipment_value_calc"]


def aggregate(df):
    """df: one row per (step s, side_idx, player) after grid filter and dedupe. Returns one row per (s, side_idx)."""
    inv = pl.col("inv_primary").fill_null(0).cast(pl.Int64)
    arm = pl.col("armor").fill_null(0).cast(pl.Int64)
    return (df.group_by(["s", "side_idx"]).agg([
        pl.col("health").fill_null(0).cast(pl.Int64).sum().alias("hp"),
        (arm > 0).cast(pl.Int64).sum().alias("armor_n"),
        arm.sum().alias("armor_sum"),
        pl.col("has_helmet").fill_null(False).cast(pl.Int64).sum().alias("helmet_n"),
        pl.col("has_defuser").fill_null(False).cast(pl.Int64).sum().alias("kit_n"),
        pl.col("is_defusing").fill_null(False).cast(pl.Int64).sum().alias("defusing"),
        (inv == AK[0]).cast(pl.Int64).sum().alias("ak"),
        inv.is_in(list(M4)).cast(pl.Int64).sum().alias("m4"),
        (inv == AWP[0]).cast(pl.Int64).sum().alias("awp"),
        inv.is_in(list(RIFLE)).cast(pl.Int64).sum().alias("rifle"),
        inv.is_in(list(SNIPER_OTHER)).cast(pl.Int64).sum().alias("sniper_other"),
        inv.is_in(list(SMG)).cast(pl.Int64).sum().alias("smg"),
        inv.is_in(list(HEAVY)).cast(pl.Int64).sum().alias("heavy"),
        (inv == 0).cast(pl.Int64).sum().alias("no_primary"),
        pl.col("inv_flashbang").fill_null(0).cast(pl.Int64).sum().alias("flash"),
        pl.col("inv_smokegrenade").fill_null(0).cast(pl.Int64).sum().alias("smoke"),
        pl.col("inv_hegrenade").fill_null(0).cast(pl.Int64).sum().alias("he"),
        (pl.col("inv_molotov").fill_null(0).cast(pl.Int64) + pl.col("inv_incgrenade").fill_null(0).cast(pl.Int64)).sum().alias("fire"),
        pl.col("inv_decoy").fill_null(0).cast(pl.Int64).sum().alias("decoy"),
        pl.col("equipment_value_calc").fill_null(0).cast(pl.Int64).sum().alias("equip"),
    ]))


def side_index(side_col):
    """CT (3) -> 0, T (2) -> 1, anything else -> -1 (dropped and counted by the caller)."""
    return (pl.when(side_col == 3).then(0).when(side_col == 2).then(1).otherwise(-1)).cast(pl.Int64)

# ---- local 1 Hz stage on the space: same rule and scope as 01_local.py, reads the staged 1 Hz caches ----
import json, sys, time, glob, os
import numpy as np
import polars as pl

T0 = time.time()


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, file=sys.stderr, flush=True)


EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
STAGE = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout_stage/"
OUTD = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
keys = json.load(open(EXP + "match_keys.json"))
split = np.load(EXP + "match_split.npy")
off = np.load(EXP + "offsets.npy").astype(np.int64)
rm = np.load(EXP + "round_match.npy").astype(np.int64)
rn = np.load(EXP + "round_num.npy").astype(np.int64)
rf = np.load(EXP + "round_freeze.npy").astype(np.int64)
nsteps = np.diff(off)
dates = [k.split("/")[1] + "-" + k.split("/")[2] + "-" + k.split("/")[3] for k in keys]
PROTECT_FROM = "2026-10-21"
in_scope = np.array([(split[i] in (0, 1)) and (dates[i] < PROTECT_FROM) for i in range(len(keys))])
log("in-scope matches", int(in_scope.sum()), "of", len(keys))
# only the keys the S3 channel does not hold (decided by the channel check) are built from local 1 Hz here
LOCAL_REQ = set(json.load(open(STAGE + "local_req_keys.json")))
req_mask = np.array([k in LOCAL_REQ for k in keys])
assert not (req_mask & (~in_scope)).any()
in_scope = in_scope & req_mask
log("local-required in-scope matches", int(in_scope.sum()))
mt = pl.DataFrame({"match_key": keys, "m": np.arange(len(keys), dtype=np.int64), "split": split.astype(np.int64),
                   "date": dates, "in_scope": in_scope})
mt_s = mt.filter(pl.col("in_scope"))
rt = pl.DataFrame({"m": rm, "round": rn, "r": np.arange(len(rm), dtype=np.int64), "freeze": rf,
                   "off_r": off[:-1], "nsteps": nsteps})
rt_s = rt.join(mt_s.select("m"), on="m", how="inner")
log("in-scope rounds", rt_s.height)
NS = int(off[-1])
arr = np.load(OUTD + "loadout_step_i16.npy", mmap_mode="r+")
arr_eq = np.load(OUTD + "loadout_equip_i32.npy", mmap_mode="r+")
assert arr.shape == (NS, 2, 19) and arr_eq.shape == (NS, 2)

files = []
for f in sorted(glob.glob(STAGE + "hz1/2026-*.parquet")):
    name = os.path.basename(f)[:10]
    if "2026-09-01" <= name <= "2026-10-20":
        files.append(("local_1hz", f))
for f in sorted(glob.glob(STAGE + "hz2024/*.parquet")):
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
    vals = np.stack([agg[cn].to_numpy() for cn in FIELDS], axis=1)
    v16 = vals[:, :-1]
    eq = vals[:, -1]
    if v16.size:
        assert v16.max() <= 32767 and v16.min() >= -32768 and eq.max() <= 2147483647
    arr[s_idx, side_idx, :] = v16.astype(np.int16)
    arr_eq[s_idx, side_idx] = eq.astype(np.int32)
    st = {"src": src, "file": os.path.basename(f), "raw_rows": n_raw, "rows_in_scope_matches": n_match,
          "rows_no_export_round": n_no_round, "rows_pre_freeze": n_pre_freeze, "rows_off_grid": n_off_grid,
          "rows_beyond_round_steps": n_beyond, "rows_bad_side": n_bad_side, "rows_kept_before_dedupe": n_pre_dedupe,
          "duplicates_dropped": n_dups, "agg_rows_written": int(len(s_idx)), "matches": int(cov.height),
          "sec": round(time.time() - t1, 1)}
    file_stats.append(st)
    log(json.dumps(st))
    del d, agg, cov
arr.flush()
arr_eq.flush()
cov_all = pl.concat(cov_parts)
cov_all = cov_all.join(mt.select("m", "match_key", "split", "date"), on="m", how="left")
cov_all.write_parquet(STAGE + "coverage_local.parquet")
json.dump(file_stats, open(STAGE + "local_file_stats.json", "w"), indent=1)
log("LOCAL STAGE DONE; coverage rows", cov_all.height, "distinct matches", cov_all["m"].n_unique())
