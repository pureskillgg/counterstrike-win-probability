import json, os, re, collections
import numpy as np
B2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
B3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout"
mk = json.load(open(os.path.join(B2, "match_keys.json")))
bk = json.load(open(os.path.join(B3, "built_match_keys.json")))
split = np.load(os.path.join(B2, "match_split.npy"))
print("type match_keys:", type(mk).__name__, "len", len(mk))
print("type built_match_keys:", type(bk).__name__, "len", len(bk) if hasattr(bk, "__len__") else "?")
if isinstance(mk, dict):
    print("match_keys dict keys sample:", list(mk.keys())[:5])
    print("match_keys first value:", str(list(mk.values())[0])[:200])
else:
    print("match_keys[0]:", str(mk[0])[:200])
if isinstance(bk, dict):
    print("built dict keys sample:", list(bk.keys())[:5])
    print("built first value:", str(list(bk.values())[0])[:300])
else:
    print("built[0]:", str(bk[0])[:300])
