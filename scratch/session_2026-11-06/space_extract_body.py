import json
import sys

import numpy as np

D = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
keys = json.load(open(D + "/match_keys.json"))
off = np.load(D + "/offsets.npy")
S = np.load(D + "/steps_alive.npy", mmap_mode="r")
L = np.load(D + "/loadout_round.npy", mmap_mode="r")
fz = np.load(D + "/round_freeze.npy")
pl_ = np.load(D + "/round_plant.npy")
rmap = np.load(D + "/round_map.npy")
rl = np.load(D + "/round_label.npy")
rn = np.load(D + "/round_num.npy")
rmatch = np.load(D + "/round_match.npy")

out = {}
for mid in MIDS:
    rs = np.where(rmatch == mid)[0]
    rounds = []
    for r in rs:
        s0, s1 = int(off[r]), int(off[r + 1])
        alive = np.asarray(S[s0:s1], dtype=np.int64).tolist()
        rounds.append({
            "r": int(r), "rn": int(rn[r]), "freeze": int(fz[r]), "plant": int(pl_[r]),
            "map": int(rmap[r]), "label": int(rl[r]),
            "loadout": np.asarray(L[r], dtype=np.float64).tolist(), "alive": alive,
        })
    out[keys[mid]] = {"mid": int(mid), "rounds": rounds}
sys.stdout.write(json.dumps(out))
