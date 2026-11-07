import json

import numpy as np

E = "/home/sagemaker-user/adam/builder/final/export_clean"
V = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
off1 = np.load(E + "/offsets.npy")
flag1 = np.load(E + "/loadout_flag.npy")
keys1 = json.load(open(E + "/match_keys.json"))
rm1 = np.load(E + "/round_match.npy")
n1 = np.diff(off1)
print("v1 rounds", len(n1), "| v1 steps", int(off1[-1]), "| v1 matches (keys)", len(keys1))
print("v1 steps_alive shape", np.load(E + "/steps_alive.npy", mmap_mode="r").shape)
print("v1 flagged rounds", int((flag1 == 1).sum()), "| steps in flagged rounds", int(n1[flag1 == 1].sum()))
print("v1 steps minus v2 steps", int(off1[-1]) - 72567700)
kept_matches = np.unique(rm1[flag1 == 0])
print("v1 matches with >=1 kept round", len(kept_matches), "| matches with zero kept rounds", len(keys1) - len(kept_matches))
print("v1 verification.json:", open(E + "/verification.json").read().replace("\n", " "))
print("v2 README.json:", open(V + "/README.json").read().replace("\n", " "))
