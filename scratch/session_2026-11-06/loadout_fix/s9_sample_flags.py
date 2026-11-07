import json, os
import numpy as np
B2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
B3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout"
mk = json.load(open(os.path.join(B2, "match_keys.json")))
bk = json.load(open(os.path.join(B3, "built_match_keys.json")))
split = np.load(os.path.join(B2, "match_split.npy"))
round_match = np.load(os.path.join(B2, "round_match.npy")).astype(np.int64)
offsets = np.load(os.path.join(B3, "offsets.npy"))
v2 = np.load(os.path.join(B2, "loadout_round.npy"), mmap_mode="r")
alive3 = np.load(os.path.join(B2, "steps_alive.npy"), mmap_mode="r")
step = np.load(os.path.join(B3, "loadout_step_i16.npy"), mmap_mode="r")
equip = np.load(os.path.join(B3, "loadout_equip_i32.npy"), mmap_mode="r")
idx_of = {k: i for i, k in enumerate(mk)}
bmask = np.zeros(len(mk), dtype=bool)
for k in bk:
    bmask[idx_of[k]] = True
cand = np.nonzero(bmask[round_match] & (split[round_match] != 2))[0]
rng = np.random.default_rng(20261105)
s_all = offsets[cand].astype(np.int64)
al_all = np.asarray(alive3[s_all]).astype(np.int64)
pos = np.nonzero((al_all[:, 0] + al_all[:, 1]) > 0)[0]
samp = np.sort(rng.choice(pos, size=25, replace=False))
rows = cand[samp]
s0 = offsets[rows].astype(np.int64)
st = np.asarray(step[s0]).astype(np.int64)
eq = np.asarray(equip[s0]).astype(np.int64)
al = np.asarray(alive3[s0]).astype(np.int64)
v2r = np.asarray(v2[rows]).astype(np.int64).reshape(len(rows), 2, 22)
f19 = np.all(st == v2r[:, :, 1:20], axis=(1, 2)) & np.all(eq == v2r[:, :, 20], axis=1)
fal = np.all(al == v2r[:, :, 0], axis=1)
print("sample rounds:", len(rows), "| seed 20261105 | rounds with alive>0 at freeze in built train/val:", len(pos))
print("rounds where all 19 per-side fields + equip agree (both sides):", int(f19.sum()), "of", len(rows))
print("rounds where alive also agrees:", int((f19 & fal).sum()), "of", len(rows))
for i in np.nonzero(~(f19 & fal))[0]:
    print("  differs: round", int(rows[i]), "19+equip agree", bool(f19[i]), "alive agree", bool(fal[i]), "steps alive", al[i].tolist(), "v2 alive", v2r[i, :, 0].tolist())
