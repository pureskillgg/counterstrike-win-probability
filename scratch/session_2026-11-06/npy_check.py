import glob
import json
import os

import numpy as np

D = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
names = json.load(open(D + "/loadout_names.json"))
maps = json.load(open(D + "/maps.json"))
readme = json.load(open(D + "/README.json"))
pos_readme = json.load(open(D + "/pos_README.json"))
print("loadout names", len(names), "maps", maps)
print("README matches/rounds/steps", readme["matches"], readme["rounds"], readme["steps"])

print("== npy headers (mmap, no writes)")
for p in sorted(glob.glob(D + "/*.npy")):
    a = np.load(p, mmap_mode="r")
    print(f"{os.path.basename(p):22s} shape {str(a.shape):22s} dtype {str(a.dtype):8s} fortran {a.flags['F_CONTIGUOUS'] and not a.flags['C_CONTIGUOUS']} bytes {os.path.getsize(p)}")

N = readme["rounds"]
L = np.load(D + "/loadout_round.npy", mmap_mode="r")
print("== loadout_round per-column min / max / mean (rounds", L.shape[0], ")")
for i, nm in enumerate(names):
    col = np.asarray(L[:, i], dtype=np.float64)
    print(f"{nm:16s} min {col.min():10.3f} max {col.max():10.3f} mean {col.mean():10.3f} nan {int(np.isnan(col).sum())}")

S = np.load(D + "/steps_alive.npy", mmap_mode="r")
off = np.load(D + "/offsets.npy")
print("== steps_alive shape", S.shape, "| offsets len", len(off), "last", int(off[-1]), "| min step count per round", int(np.diff(off).min()), "max", int(np.diff(off).max()))
print("steps_alive col0 min/max", int(S[:, 0].min()), int(S[:, 0].max()), "| col1 min/max", int(S[:, 1].min()), int(S[:, 1].max()))

first = off[:-1]
ca = L[:, names.index("CT_alive")].astype(np.int64)
ta = L[:, names.index("T_alive")].astype(np.int64)
s0c = np.asarray(S[first, 0]).astype(np.int64)
s0t = np.asarray(S[first, 1]).astype(np.int64)
print("== cross-check: steps_alive at step 0 vs loadout CT_alive / T_alive (all rounds)")
print("CT mismatches", int((s0c != ca).sum()), "| T mismatches", int((s0t != ta).sum()))
print("loadout CT_alive value counts", dict(zip(*[x.tolist() for x in np.unique(ca, return_counts=True)])))
print("loadout T_alive value counts", dict(zip(*[x.tolist() for x in np.unique(ta, return_counts=True)])))

flag = np.load(D + "/loadout_flag.npy")
print("loadout_flag unique", np.unique(flag).tolist())

fz = np.load(D + "/round_freeze.npy")
pl_ = np.load(D + "/round_plant.npy")
rl = np.load(D + "/round_label.npy")
rm_ = np.load(D + "/round_map.npy")
rn = np.load(D + "/round_num.npy")
rmatch = np.load(D + "/round_match.npy")
split = np.load(D + "/match_split.npy")
print("round_freeze min/max", int(fz.min()), int(fz.max()), "| round_plant min/max", int(pl_.min()), int(pl_.max()), "| planted rounds (plant>0)", int((pl_ > 0).sum()), "| plant==-1", int((pl_ == -1).sum()))
print("round_label unique", np.unique(rl).tolist(), "counts", np.bincount(rl.astype(np.int64)).tolist())
print("round_map unique", np.unique(rm_).tolist(), "counts", np.bincount(rm_.astype(np.int64)).tolist())
print("round_num min/max", int(rn.min()), int(rn.max()))
print("round_match min/max", int(rmatch.min()), int(rmatch.max()), "unique", len(np.unique(rmatch)), "| match_split unique", np.unique(split).tolist(), np.bincount(split.astype(np.int64)).tolist())
print("rounds per match check: unique matches in round_match", len(np.unique(rmatch)), "vs match_keys", len(json.load(open(D + "/match_keys.json"))))

pi8 = np.load(D + "/pos_int8.npy", mmap_mode="r")
pu16 = np.load(D + "/pos_uint16.npy", mmap_mode="r")
pu8 = np.load(D + "/pos_uint8.npy", mmap_mode="r")
pf16 = np.load(D + "/pos_float16.npy", mmap_mode="r")
po = np.load(D + "/pos_offsets.npy")
cnt = (np.diff(off) + 1) // 2
print("== position arrays")
print("int8", pi8.shape, pi8.dtype, "cols", len(pos_readme["int8"]))
print("uint16", pu16.shape, pu16.dtype, "cols", len(pos_readme["uint16"]))
print("uint8", pu8.shape, pu8.dtype, "cols", len(pos_readme["uint8_tenth_seconds"]))
print("float16", pf16.shape, pf16.dtype, "cols 1")
print("pos_offsets last", int(po[-1]), "matches steps_even", pos_readme["steps_even"], "| pos_offsets == cumsum(ceil(n/2))", bool(np.array_equal(np.diff(po), cnt)))
total_cols = len(pos_readme["int8"]) + len(pos_readme["uint16"]) + len(pos_readme["uint8_tenth_seconds"]) + len(pos_readme["float16"])
print("position feature count", total_cols)
for label, arr, cols, sentinel in [("int8", pi8, pos_readme["int8"], -1), ("uint16", pu16, pos_readme["uint16"], 65535), ("uint8", pu8, pos_readme["uint8_tenth_seconds"], 255)]:
    for i, c in enumerate(cols):
        col = np.asarray(arr[:, i])
        miss = int((col == sentinel).sum())
        ok = col[col != sentinel]
        exp_miss = pos_readme["missing_per_feature"][c]
        print(f"{label:6s} {c:22s} min {ok.min() if ok.size else 'na'} max {ok.max() if ok.size else 'na'} sentinel-count {miss} readme-missing {exp_miss} {'OK' if miss == exp_miss else 'DIFF'}")
col = np.asarray(pf16[:])
nan = int(np.isnan(col.astype(np.float32)).sum())
print("defuse_slack min/max (non-nan)", float(np.nanmin(col)), float(np.nanmax(col)), "nan", nan, "readme-missing", pos_readme["missing_per_feature"]["defuse_slack"])
