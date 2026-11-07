import json
import numpy as np

A = "/home/sagemaker-user/adam/builder/final/export_clean"
B = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
oa, ob = np.load(A + "/offsets.npy"), np.load(B + "/offsets.npy")
print("offsets len", len(oa), len(ob), "equal", np.array_equal(oa, ob), "last", int(oa[-1]), int(ob[-1]))
ka, kb = json.load(open(A + "/match_keys.json")), json.load(open(B + "/match_keys.json"))
print("match_keys equal", ka == kb, len(ka), len(kb))
for f in ("round_freeze.npy", "round_match.npy", "round_num.npy", "round_plant.npy"):
    print(f, "equal", np.array_equal(np.load(A + "/" + f), np.load(B + "/" + f)))
print("steps total (export_clean)", int(np.diff(oa).sum()), "rounds", len(np.load(A + "/round_match.npy")))
