import json, os
import numpy as np
B2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
B3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout"
mk = json.load(open(os.path.join(B2, "match_keys.json")))
bk = json.load(open(os.path.join(B3, "built_match_keys.json")))
split = np.load(os.path.join(B2, "match_split.npy"))
names = json.load(open(os.path.join(B2, "loadout_names.json")))
round_match = np.load(os.path.join(B2, "round_match.npy"))
round_freeze = np.load(os.path.join(B2, "round_freeze.npy"))
offsets = np.load(os.path.join(B3, "offsets.npy"))
v2 = np.load(os.path.join(B2, "loadout_round.npy"), mmap_mode="r")
alive3 = np.load(os.path.join(B2, "steps_alive.npy"), mmap_mode="r")
step = np.load(os.path.join(B3, "loadout_step_i16.npy"), mmap_mode="r")
equip = np.load(os.path.join(B3, "loadout_equip_i32.npy"), mmap_mode="r")
print("shapes: v2 loadout_round", v2.shape, "| v3 step", step.shape, "| v3 equip", equip.shape, "| steps_alive", alive3.shape)
print("names count", len(names), "per side", len(names)//2)
v3_fields = ["hp","armor_n","armor_sum","helmet_n","kit_n","defusing","ak","m4","awp","rifle","sniper_other","smg","heavy","no_primary","flash","smoke","he","fire","decoy"]
# mapping check: v3 field j -> v2 column 1+j within each side block of 22
ok = True
for side, pre in ((0, "CT_"), (1, "T_")):
    for j, f in enumerate(v3_fields):
        if names[side*22 + 1 + j] != pre + f:
            ok = False; print("MAP MISMATCH", side, j, names[side*22+1+j], pre+f)
    if names[side*22 + 20] != pre + "equip": ok = False; print("equip map mismatch", side)
    if names[side*22 + 0] != pre + "alive": ok = False; print("alive map mismatch", side)
print("name mapping v3 -> v2 columns consistent:", ok)
idx_of = {k: i for i, k in enumerate(mk)}
bmask = np.zeros(len(mk), dtype=bool)
for k in bk:
    bmask[idx_of[k]] = True
rm = round_match.astype(np.int64)
cand = np.nonzero(bmask[rm] & (split[rm] != 2))[0]
nsteps = offsets[cand + 1] - offsets[cand]
print("rounds total", len(round_match), "| rounds in built matches, split 0/1:", len(cand), "| with >=1 step:", int((nsteps >= 1).sum()))
cand = cand[nsteps >= 1]
s0 = offsets[cand].astype(np.int64)
R = len(cand)
st = np.asarray(step[s0]).astype(np.int64)        # (R,2,19) freeze-second per-side fields
eq = np.asarray(equip[s0]).astype(np.int64)       # (R,2)
al = np.asarray(alive3[s0]).astype(np.int64)      # (R,2)
v2r = np.asarray(v2[cand]).astype(np.int64).reshape(R, 2, 22)
print("compared rounds R =", R)
tot_cells = 0; tot_eq = 0
field_mis = {}
row_ok = np.ones(R, dtype=bool)
for side, pre in ((0, "CT"), (1, "T")):
    for j, f in enumerate(v3_fields):
        eqm = st[:, side, j] == v2r[:, side, 1 + j]
        field_mis[f"{pre}_{f}"] = int((~eqm).sum())
        row_ok &= eqm
        tot_cells += R; tot_eq += int(eqm.sum())
    eqm = eq[:, side] == v2r[:, side, 20]
    field_mis[f"{pre}_equip"] = int((~eqm).sum()); row_ok &= eqm; tot_cells += R; tot_eq += int(eqm.sum())
    eqm = al[:, side] == v2r[:, side, 0]
    field_mis[f"{pre}_alive(steps_alive vs v2 alive)"] = int((~eqm).sum()); row_ok &= eqm; tot_cells += R; tot_eq += int(eqm.sum())
print("cells compared:", tot_cells, "| equal:", tot_eq, "| mismatched:", tot_cells - tot_eq)
print("rounds with every compared field equal:", int(row_ok.sum()), "of", R)
print("mismatch counts by field (nonzero only):", {k: v for k, v in field_mis.items() if v})
print("zero-mismatch fields:", sum(1 for v in field_mis.values() if v == 0), "of", len(field_mis))
alive_pos = int(((al[:, 0] + al[:, 1]) > 0).sum())
print("rounds with alive>0 at freeze (either side):", alive_pos)
print("v2 money column (not in v3, not compared): min", float(np.min(v2r[:, :, 21])), "max", float(np.max(v2r[:, :, 21])))
bad = np.nonzero(~row_ok)[0]
for b in bad[:5]:
    r = int(cand[b])
    print("MISMATCH round", r, "offset", int(s0[b]), "CT v3", st[b,0].tolist(), "eq", int(eq[b,0]), "al", int(al[b,0]), "| CT v2", v2r[b,0].tolist())
    print("   T v3", st[b,1].tolist(), "eq", int(eq[b,1]), "al", int(al[b,1]), "| T v2", v2r[b,1].tolist())
# sample of rounds with per-second loadouts (alive>0), show side by side
rng = np.random.default_rng(20261105)
pos = np.nonzero((al[:, 0] + al[:, 1]) > 0)[0]
samp = rng.choice(pos, size=min(25, len(pos)), replace=False)
print("SAMPLE (25 random built rounds with alive>0; fields: hp,armor_n,armor_sum,helmet_n,kit_n,defusing,ak,m4,awp,rifle,sniper_other,smg,heavy,no_primary,flash,smoke,he,fire,decoy | equip | alive)")
for b in samp:
    r = int(cand[b])
    same = bool(row_ok[b])
    print(f"round {r} freeze_tick {int(round_freeze[r])} offset {int(s0[b])} agree={same}")
    print("  CT v3", st[b,0].tolist(), eq[b,0], al[b,0])
    print("  CT v2", v2r[b,0,1:20].tolist(), v2r[b,0,20], v2r[b,0,0])
    print("  T  v3", st[b,1].tolist(), eq[b,1], al[b,1])
    print("  T  v2", v2r[b,1,1:20].tolist(), v2r[b,1,20], v2r[b,1,0])
