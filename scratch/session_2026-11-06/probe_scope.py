import json, collections
import numpy as np
EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
keys = json.load(open(EXP + "match_keys.json"))
print("match_keys", len(keys), "unique", len(set(keys)))
for f in ["round_match", "round_num", "round_freeze", "round_label", "round_map", "round_plant", "loadout_round", "offsets"]:
    try:
        a = np.load(EXP + f + ".npy", mmap_mode="r")
        print(f, a.shape, a.dtype)
    except Exception as e:
        print(f, "ERR", e)
rm = np.load(EXP + "round_match.npy", mmap_mode="r")
print("round_match max", int(rm.max()), "min", int(rm.min()), "distinct", len(np.unique(rm)))
