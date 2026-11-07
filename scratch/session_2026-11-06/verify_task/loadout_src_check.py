import csv
import json
import numpy as np

V2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"

keys = json.load(open(V2 + "match_keys.json"))
rm = np.load(V2 + "round_match.npy").astype(np.int64)
status = {}
with open(V4 + "match_loadout_status.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        status[r["match_key"]] = r["v4_batch"] or "v3_built"
is_new_match = np.array([status[k] != "v3_built" for k in keys])
is_new_round = is_new_match[rm]
names = json.load(open(V4 + "loadout_names.json"))
v2 = np.asarray(np.load(V2 + "loadout_round.npy", mmap_mode="r"))
nz = np.any(v2 != 0, axis=1)
print("v2 loadout_round nonzero rows, new rounds:", int((nz & is_new_round).sum()), "of", int(is_new_round.sum()),
      "| built rounds:", int((nz & ~is_new_round).sum()), "of", int((~is_new_round).sum()))
pick = ["CT_alive", "CT_hp", "CT_money", "CT_rifle", "CT_equip", "T_alive", "T_hp", "T_money", "T_rifle", "T_equip"]
idx = [names.index(p) for p in pick]
print("mean, new rounds (v2):", dict(zip(pick, np.round(v2[is_new_round][:, idx].mean(axis=0), 2).tolist())))
print("mean, built rounds (v2):", dict(zip(pick, np.round(v2[~is_new_round][:, idx].mean(axis=0), 2).tolist())))
print("CT_alive value counts, new rounds:", np.unique(v2[is_new_round][:, names.index("CT_alive")], return_counts=True)[0].tolist(),
      np.unique(v2[is_new_round][:, names.index("CT_alive")], return_counts=True)[1].tolist())
print("CT_alive value counts, built rounds:", np.unique(v2[~is_new_round][:, names.index("CT_alive")], return_counts=True)[0].tolist(),
      np.unique(v2[~is_new_round][:, names.index("CT_alive")], return_counts=True)[1].tolist())
print("DONE", flush=True)
