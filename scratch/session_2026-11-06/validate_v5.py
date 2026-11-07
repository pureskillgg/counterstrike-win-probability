"""Validate export_clean_v5 against export_clean_v4.

Pass criteria:
  - every changed step (positions, per-second loadouts, equipment) and every changed freeze row lies inside a target round
    (525 train/val rounds of the 41 affected train/val matches);
  - target freeze rows: 5 CT and 5 T alive (loadout_round);
  - the step index (offsets) and pos_offsets are identical to v4.
Read-only on v4 and v5. Writes only validate_v5.json under fixes/fix332/v5_build/.
"""
import json
import numpy as np
import polars as pl

V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
V5 = "/home/sagemaker-user/adam/builder/final/export_clean_v5/"
B = "/home/sagemaker-user/adam/builder/fixes/fix332/v5_build/"
BLK = 4_000_000

off = np.load(V4 + "offsets.npy").astype(np.int64)
S = int(off[-1])
rm = np.load(V4 + "round_match.npy").astype(np.int64)
rn = np.load(V4 + "round_num.npy").astype(np.int64)
keys = json.load(open(V4 + "match_keys.json"))
kid = {k: i for i, k in enumerate(keys)}
ovr = pl.read_parquet(B + "ovr_trainval.parquet").select("match_key", "round").unique()
target_pairs = set((kid[m], r) for m, r in zip(ovr["match_key"].to_list(), ovr["round"].to_list()))
R = len(rm)
is_target = np.zeros(R, bool)
for i in range(R):
    if (int(rm[i]), int(rn[i])) in target_pairs:
        is_target[i] = True
print("target rounds in export", int(is_target.sum()), "expected", len(target_pairs), flush=True)
assert int(is_target.sum()) == len(target_pairs)

res = {"target_rounds": int(is_target.sum()), "steps": S, "arrays": {}}
assert np.array_equal(off, np.load(V5 + "offsets.npy").astype(np.int64)), "offsets differ"
assert np.array_equal(np.load(V4 + "pos_offsets.npy").astype(np.int64), np.load(V5 + "pos_offsets.npy").astype(np.int64)), "pos_offsets differ"
print("offsets and pos_offsets identical", flush=True)


def step_to_round(s):
    return np.searchsorted(off, s, side="right") - 1


def compare_steps(name, a4, a5, flat=True):
    """a4, a5: (S, ...) arrays. Counts changed steps and cells; flags changes outside target rounds."""
    changed_steps = 0
    changed_cells = 0
    outside_steps = 0
    outside_cells = 0
    target_changed_steps = 0
    for b0 in range(0, S, BLK):
        b1 = min(S, b0 + BLK)
        x = np.asarray(a4[b0:b1])
        y = np.asarray(a5[b0:b1])
        if x.dtype.kind == "f":
            neq = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        else:
            neq = x != y
        if neq.ndim > 1:
            cells_per_step = neq.reshape(neq.shape[0], -1).sum(1)
        else:
            cells_per_step = neq.astype(np.int64)
        idx = np.nonzero(cells_per_step)[0]
        if idx.size:
            steps = idx + b0
            rounds = step_to_round(steps)
            tgt = is_target[rounds]
            changed_steps += int(idx.size)
            changed_cells += int(cells_per_step[idx].sum())
            target_changed_steps += int(tgt.sum())
            outside_steps += int((~tgt).sum())
            outside_cells += int(cells_per_step[idx][~tgt].sum())
    res["arrays"][name] = {"changed_steps": changed_steps, "changed_cells": changed_cells,
                           "changed_steps_in_target_rounds": target_changed_steps,
                           "changed_steps_outside_target_rounds": outside_steps,
                           "changed_cells_outside_target_rounds": outside_cells}
    print(name, res["arrays"][name], flush=True)
    assert outside_steps == 0, f"{name}: {outside_steps} changed steps outside target rounds"


for fname in ["pos_int8.npy", "pos_uint16.npy", "pos_uint8.npy", "pos_float16.npy"]:
    a4 = np.load(V4 + "pos_v4/" + fname, mmap_mode="r")
    a5 = np.load(V5 + "pos_v5/" + fname, mmap_mode="r")
    assert a4.shape == a5.shape, (fname, a4.shape, a5.shape)
    compare_steps("pos_v/" + fname, a4, a5)

a4 = np.load(V4 + "loadout_step_i16.npy", mmap_mode="r")
a5 = np.load(V5 + "loadout_step_i16.npy", mmap_mode="r")
compare_steps("loadout_step_i16", a4, a5)
e4 = np.load(V4 + "loadout_equip_i32.npy", mmap_mode="r")
e5 = np.load(V5 + "loadout_equip_i32.npy", mmap_mode="r")
compare_steps("loadout_equip_i32", e4, e5)

# freeze snapshot
names = json.load(open(V4 + "loadout_names.json"))
ia_ct, ia_t = names.index("CT_alive"), names.index("T_alive")
L4 = np.load(V4 + "loadout_round.npy", mmap_mode="r")
L5 = np.load(V5 + "loadout_round.npy", mmap_mode="r")
x = np.asarray(L4)
y = np.asarray(L5)
changed_rows = np.nonzero(np.any(x != y, axis=1))[0]
outside = int((~is_target[changed_rows]).sum())
alive_bad = 0
for r in np.nonzero(is_target)[0]:
    if not (y[r, ia_ct] == 5 and y[r, ia_t] == 5):
        alive_bad += 1
res["freeze_snapshot"] = {"changed_rows": int(changed_rows.size), "changed_rows_outside_target": outside,
                          "target_rows_not_5v5": alive_bad,
                          "target_rows": int(is_target.sum())}
print("freeze snapshot", res["freeze_snapshot"], flush=True)
assert outside == 0 and alive_bad == 0

json.dump(res, open(B + "validate_v5.json", "w"), indent=1)
print("VALIDATE_PASS", flush=True)
