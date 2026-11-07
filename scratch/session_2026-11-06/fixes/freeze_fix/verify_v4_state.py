"""Read-only check of export_clean_v4 loadout step arrays against export_clean_v3_loadout (no writes).

Counts steps where v4 differs from v3, split by (a) the 23 freeze-investigated matches and the 283 confirmed rounds, (b) every other
match (expected: the coverage build's 10,805,134 steps, from scripts/verify_v4.json). Also checks the status CSV and the
loadout_round.npy patch against v2.
"""
import json
import numpy as np
import polars as pl

V2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
V3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
OUT = FX + "verify_v4_state.json"
keys = json.load(open(V2 + "match_keys.json"))
kid = {k: i for i, k in enumerate(keys)}
off = np.load(V2 + "offsets.npy").astype(np.int64)
rm = np.load(V2 + "round_match.npy").astype(np.int64)
rn = np.load(V2 + "round_num.npy").astype(np.int64)
NS = int(off[-1])
inv = set(json.load(open(FX + "inv_matches.json")))
inv_ids = np.array(sorted(kid[k] for k in inv), dtype=np.int64)
fx = pl.read_csv(FX + "override_rounds.csv").filter(pl.col("status") == "fixed").select("match_key", "round")
fixed_ridx = set()
for mk, rd in fx.rows():
    m = kid[mk]
    idx = np.nonzero((rm == m) & (rn == int(rd)))[0]
    assert idx.size == 1, (mk, rd, idx.size)
    fixed_ridx.add(int(idx[0]))
fixed_ridx_arr = np.array(sorted(fixed_ridx), dtype=np.int64)
is_fixed_round = np.zeros(len(rm), dtype=bool)
is_fixed_round[fixed_ridx_arr] = True
is_inv_round = np.isin(rm, inv_ids)
a16 = np.load(V3 + "loadout_step_i16.npy", mmap_mode="r")
b16 = np.load(V4 + "loadout_step_i16.npy", mmap_mode="r")
ae = np.load(V3 + "loadout_equip_i32.npy", mmap_mode="r")
be = np.load(V4 + "loadout_equip_i32.npy", mmap_mode="r")
assert a16.shape == b16.shape == (NS, 2, 19)
cnt = {"steps_changed_inv_fixed_rounds": 0, "steps_changed_inv_other_rounds": 0, "steps_changed_other_matches": 0,
       "steps_in_inv_fixed_rounds": 0, "steps_in_inv_other_rounds": 0}
BLOCK = 4_000_000
for s0 in range(0, NS, BLOCK):
    s1 = min(NS, s0 + BLOCK)
    x = np.asarray(a16[s0:s1])
    y = np.asarray(b16[s0:s1])
    xe = np.asarray(ae[s0:s1])
    ye = np.asarray(be[s0:s1])
    changed = np.any(x != y, axis=(1, 2)) | np.any(xe != ye, axis=1)
    steps = np.arange(s0, s1, dtype=np.int64)
    r = np.searchsorted(off, steps, side="right") - 1
    fixed_r = is_fixed_round[r]
    inv_r = is_inv_round[r]
    cnt["steps_changed_inv_fixed_rounds"] += int(np.sum(changed & fixed_r))
    cnt["steps_changed_inv_other_rounds"] += int(np.sum(changed & inv_r & ~fixed_r))
    cnt["steps_changed_other_matches"] += int(np.sum(changed & ~inv_r))
    cnt["steps_in_inv_fixed_rounds"] += int(np.sum(fixed_r))
    cnt["steps_in_inv_other_rounds"] += int(np.sum(inv_r & ~fixed_r))
    print(f"block {s0}-{s1} done", flush=True)
# loadout_round.npy vs v2
v2r = np.load(V2 + "loadout_round.npy", mmap_mode="r")
v4r = np.load(V4 + "loadout_round.npy", mmap_mode="r")
round_changed = np.nonzero(np.any(np.asarray(v2r) != np.asarray(v4r), axis=1))[0]
out = {"step_counts": cnt, "fixed_rounds": len(fixed_ridx), "inv_matches": len(inv),
       "loadout_round_rows_changed_vs_v2": int(round_changed.size),
       "loadout_round_rows_changed_all_fixed": bool(set(round_changed.tolist()) <= fixed_ridx),
       "loadout_round_rows_changed_cells": int(np.sum(np.asarray(v2r) != np.asarray(v4r)))}
st4 = pl.read_csv(V4 + "match_loadout_status.csv", infer_schema_length=0)
st3 = pl.read_csv(V3 + "match_loadout_status.csv", infer_schema_length=0)
out["status_csv_v4_columns"] = st4.columns
out["status_csv_v4_rows"] = st4.height
out["status_csv_v3_rows"] = st3.height
out["status_csv_build_status_counts"] = {str(k): int(v) for k, v in st4.group_by("build_status").len().rows()}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1), flush=True)
print("VERIFY DONE", flush=True)
