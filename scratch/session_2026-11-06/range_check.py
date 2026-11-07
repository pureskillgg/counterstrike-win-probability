import json
import numpy as np
D = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
names = json.load(open(D + "/loadout_names.json"))
L = np.load(D + "/loadout_round.npy", mmap_mode="r")
rn = np.load(D + "/round_num.npy")
def col(n): return np.asarray(L[:, names.index(n)]).astype(np.float64)
tk = col("T_kit_n")
print("rounds with T_kit_n > 0:", int((tk > 0).sum()), "| max", tk.max())
ca, ta = col("CT_alive"), col("T_alive")
print("rounds with CT_alive < 5 at freeze:", int((ca < 5).sum()), "| T_alive < 5:", int((ta < 5).sum()), "| both 5:", int(((ca == 5) & (ta == 5)).sum()))
print("rounds with CT_alive < 5 or T_alive < 5:", int(((ca < 5) | (ta < 5)).sum()))
print("round_num > 24:", int((rn > 24).sum()), "| round_num > 30:", int((rn > 30).sum()), "| max", int(rn.max()))
