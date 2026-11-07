"""v2-side freeze-tick fix (runs on the space).

Recomputes the per-round freeze-tick loadout (export_clean_v2 loadout_round.npy, 22 features per side, CT block then T block) for the
23 investigated matches, with the v2 semantics of loadout_build2.py (rows at tick == freeze with health > 0; side from the roster;
alive players without a side dropped). Two rosters:
  OLD : channel roster, first team code per (round, player)            -- used to validate against the existing v2 array
  NEW : OLD, but the confirmed rounds (override_rows.parquet) take the side from place_name at freeze, and their unrostered alive player is added
Validation: OLD must reproduce the existing v2 rows on the rounds that the override does not touch.
Patch: only the confirmed rounds, and only where NEW differs from the existing v2 row. Written into export_clean_v4/loadout_round.npy
(copied from v2 first if missing). export_clean_v2 is never written.
"""
import json
import io
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
V2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
OUT = FX + "v2_fix_out/"
os.makedirs(OUT, exist_ok=True)
CHUNKS = ["c%03d" % i for i in range(0, 63)]
RIFLE = (7, 8, 10, 13, 16, 39, 60)
SNIPER_OTHER = (11, 38, 40)
SMG = (17, 19, 23, 24, 26, 33, 34)
HEAVY = (14, 25, 27, 28, 29, 35)
FEATS = ["alive", "hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
         "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy", "equip", "money"]
names = json.load(open(V2 + "loadout_names.json"))
assert names == [f"{s}_{f}" for s in ("CT", "T") for f in FEATS], "v2 name order"


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, flush=True)


s3 = boto3.client("s3", config=Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "standard"}))
POOL = ThreadPoolExecutor(12)
keys = json.load(open(V2 + "match_keys.json"))
kid = {k: i for i, k in enumerate(keys)}
rm = np.load(V2 + "round_match.npy").astype(np.int64)
rn = np.load(V2 + "round_num.npy").astype(np.int64)
rf = np.load(V2 + "round_freeze.npy").astype(np.int64)
inv = json.load(open(FX + "inv_matches.json"))
inv_ids = set(kid[k] for k in inv)
ridx = {}
for r in range(len(rm)):
    if int(rm[r]) in inv_ids:
        ridx[(keys[int(rm[r])], int(rn[r]))] = r
rd = pl.DataFrame({"match_key": [k for (k, _) in ridx], "round": [n for (_, n) in ridx],
                   "ridx": list(ridx.values()), "freeze": [int(rf[r]) for r in ridx.values()]},
                  schema={"match_key": pl.String, "round": pl.Int64, "ridx": pl.Int64, "freeze": pl.Int64})
log("rounds of investigated matches", rd.height)
ovr = pl.read_parquet(FX + "override_rows.parquet").select(["match_key", "round", "player_id", "side"]).with_columns(
    pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("side").cast(pl.Int64))
fixed_status = pl.read_csv(FX + "override_rounds.csv").select(["match_key", "round", "status"]).with_columns(pl.col("round").cast(pl.Int64))
fixed_set = set(map(tuple, fixed_status.filter(pl.col("status") == "fixed").select("match_key", "round").rows()))
log("fixed rounds", len(fixed_set))


def list_pages(c, kind):
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/{c}/tome/csds/{kind}."):
        for o in page.get("Contents", []):
            d, n = o["Key"].rsplit("/", 1)
            if d.rsplit("/", 1)[-1].startswith(kind + ".") and n.startswith("dataframe_"):
                out.append(o["Key"])
    return sorted(out)


def read_pages(pkeys, columns):
    frames = []
    for b in POOL.map(lambda k: s3.get_object(Bucket=BUCKET, Key=k)["Body"].read(), pkeys):
        frames.append(pl.read_parquet(io.BytesIO(b)).select(columns))
    return frames


MATCHES = set(inv)
ST_COLS = ["match_key", "round", "tick", "player_id", "health", "armor", "has_helmet", "has_defuser", "is_defusing", "money",
           "equipment_value_calc", "inv_primary", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov", "inv_incgrenade", "inv_decoy"]
INFO_COLS = ["match_key", "round", "player_id", "team_code"]
st_parts, info_parts = [], []
for c in CHUNKS:
    sk = list_pages(c, "basic_player_status")
    if not sk:
        continue
    ik = list_pages(c, "basic_player_info")
    inf = pl.concat(read_pages(ik, INFO_COLS)).filter(pl.col("match_key").is_in(list(MATCHES)))
    if inf.height:
        info_parts.append(inf.with_columns(pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("team_code").cast(pl.Int64)))
    st = pl.concat(read_pages(sk, ST_COLS)).filter(pl.col("match_key").is_in(list(MATCHES)))
    if st.height:
        st = st.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
        st = st.join(rd.select("match_key", "round", "freeze"), on=["match_key", "round"], how="inner").filter(
            (pl.col("tick") == pl.col("freeze")) & (pl.col("health") > 0))
        st_parts.append(st.drop("tick", "freeze"))
    log(c, "info rows", inf.height, "freeze alive rows so far", sum(p.height for p in st_parts))
S = pl.concat(st_parts)  # no dedupe: v2 (loadout_build2.py) counts every freeze row
I = pl.concat(info_parts).unique(subset=["match_key", "round", "player_id", "team_code"], keep="first")
old_roster = I.group_by(["match_key", "round", "player_id"]).agg(pl.col("team_code").first().alias("team_code"))
S.write_parquet(OUT + "freeze_alive_rows.parquet")
log("freeze alive players", S.height, "| roster players", old_roster.height)


def side_agg(st_with_side):
    inv_ = pl.col("inv_primary")
    return st_with_side.group_by(["match_key", "round", "side"]).agg(
        pl.len().alias("alive"),
        pl.col("health").sum().alias("hp"),
        (pl.col("armor") > 0).cast(pl.Int64).sum().alias("armor_n"),
        pl.col("armor").sum().alias("armor_sum"),
        pl.col("has_helmet").cast(pl.Int64).sum().alias("helmet_n"),
        pl.col("has_defuser").cast(pl.Int64).sum().alias("kit_n"),
        pl.col("is_defusing").cast(pl.Int64).sum().alias("defusing"),
        inv_.is_in([7]).cast(pl.Int64).sum().alias("ak"),
        inv_.is_in([16, 60]).cast(pl.Int64).sum().alias("m4"),
        inv_.is_in([9]).cast(pl.Int64).sum().alias("awp"),
        inv_.is_in(list(RIFLE)).cast(pl.Int64).sum().alias("rifle"),
        inv_.is_in(list(SNIPER_OTHER)).cast(pl.Int64).sum().alias("sniper_other"),
        inv_.is_in(list(SMG)).cast(pl.Int64).sum().alias("smg"),
        inv_.is_in(list(HEAVY)).cast(pl.Int64).sum().alias("heavy"),
        (inv_ == 0).cast(pl.Int64).sum().alias("no_primary"),
        pl.col("inv_flashbang").sum().alias("flash"),
        pl.col("inv_smokegrenade").sum().alias("smoke"),
        pl.col("inv_hegrenade").sum().alias("he"),
        (pl.col("inv_molotov") + pl.col("inv_incgrenade")).sum().alias("fire"),
        pl.col("inv_decoy").sum().alias("decoy"),
        pl.col("equipment_value_calc").sum().alias("equip"),
        pl.col("money").sum().alias("money"),
    )


def features(roster):
    j = S.join(roster, on=["match_key", "round", "player_id"], how="inner").with_columns(
        pl.col("team_code").alias("side")).filter(pl.col("side").is_in([2, 3]))
    return side_agg(j)


# OLD roster: channel first code, unrostered dropped (v2 behaviour, first instead of keep-last)
agg_old = features(old_roster)
# NEW roster: confirmed rounds take the override sides (all their alive players), other rounds unchanged
ovr_new = ovr.select("match_key", "round", "player_id", pl.col("side").alias("team_code"))
keep_old = old_roster.join(ovr_new.select("match_key", "round").unique(), on=["match_key", "round"], how="anti")
new_roster = pl.concat([ovr_new, keep_old], how="vertical_relaxed")
agg_new = features(new_roster)


def to_matrix(agg):
    M = np.zeros((len(rm), len(names)), dtype=np.float64)
    have = np.zeros(len(rm), dtype=bool)
    rows = agg.select(["match_key", "round", "side"] + FEATS).rows()
    for mk, rnd, sd, *vals in rows:
        r = ridx.get((mk, int(rnd)))
        if r is None:
            continue
        base = 0 if sd == 3 else len(FEATS)
        M[r, base:base + len(FEATS)] = np.asarray(vals, dtype=np.float64)
        have[r] = True
    return M, have


M_old, have_old = to_matrix(agg_old)
M_new, have_new = to_matrix(agg_new)
V = np.load(V2 + "loadout_round.npy", mmap_mode="r")
existing = np.asarray(V[sorted(ridx.values())], dtype=np.float64)
idx_sorted = sorted(ridx.values())
pos_of = {r: i for i, r in enumerate(idx_sorted)}
rows_fixed, rows_other = [], []
rec = []
for (mk, rnd), r in ridx.items():
    is_fixed = (mk, rnd) in fixed_set
    old_ok = bool(np.allclose(M_old[r], existing[pos_of[r]], atol=1e-3))
    new_eq_exist = bool(np.allclose(M_new[r], existing[pos_of[r]], atol=1e-3))
    (rows_fixed if is_fixed else rows_other).append((mk, rnd, r, old_ok))
    rec.append({"match_key": mk, "round": rnd, "ridx": r, "fixed": is_fixed, "old_recompute_equals_v2": old_ok,
                "new_equals_v2": new_eq_exist, "alive_new_CT": float(M_new[r, 0]), "alive_new_T": float(M_new[r, len(FEATS)])})
pl.DataFrame(rec).write_csv(OUT + "v2_round_check.csv")
n_other = len(rows_other)
print("non-fixed rounds", n_other, "| old recompute equals v2", sum(1 for x in rows_other if x[3]), flush=True)
print("fixed rounds", len(rows_fixed), "| old recompute equals v2", sum(1 for x in rows_fixed if x[3]), flush=True)
gate_bad = [x for x in rows_fixed if not (M_new[x[2], 0] == 5 and M_new[x[2], len(FEATS)] == 5)]
print("fixed rounds failing 5 alive per side gate (alive CT/T from new recompute)", len(gate_bad), flush=True)

# patch v4 loadout_round.npy: only fixed rounds, only where the new value differs from v2
if not os.path.exists(V4 + "loadout_round.npy"):
    import shutil
    shutil.copyfile(V2 + "loadout_round.npy", V4 + "loadout_round.npy")
Vw = np.load(V4 + "loadout_round.npy", mmap_mode="r+")
chg = []
n_patch = 0
for mk, rnd, r, _ in rows_fixed:
    newv = M_new[r].astype(np.float32)
    oldv = np.asarray(Vw[r]).astype(np.float32)
    if not np.allclose(newv, oldv, atol=1e-3):
        for j2, nm in enumerate(names):
            if abs(float(newv[j2]) - float(oldv[j2])) > 1e-3:
                chg.append({"match_key": mk, "round": rnd, "ridx": r, "feature": nm, "old": float(oldv[j2]), "new": float(newv[j2])})
        Vw[r] = newv
        n_patch += 1
Vw.flush()
pl.DataFrame(chg).write_csv(OUT + "v2_changed_cells.csv") if chg else None
log("patched rounds", n_patch, "| changed cells", len(chg))
json.dump({"rounds_checked": len(ridx), "fixed_rounds": len(rows_fixed), "patched_rounds": n_patch, "changed_cells": len(chg),
           "gate_failures": len(gate_bad), "non_fixed_old_recompute_equal": sum(1 for x in rows_other if x[3]),
           "non_fixed_total": n_other, "fixed_old_recompute_equal": sum(1 for x in rows_fixed if x[3]), "sec": round(time.time() - T0, 1)},
          open(OUT + "v2_fix_summary.json", "w"), indent=1)
log("done")
