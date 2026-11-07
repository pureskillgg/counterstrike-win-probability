import json, sys, collections
import numpy as np
import polars as pl
S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
E = S + "/export"
keys = json.load(open(S + "/match_keys.json"))
split = np.load(E + "/match_split.npy")
off = np.load(S + "/offsets.npy")
rm = np.load(S + "/round_match.npy")
rn = np.load(S + "/round_num.npy")
rf = np.load(S + "/round_freeze.npy")
rp = np.load(E + "/round_plant.npy")
print("matches", len(keys), "rounds", len(rm), "offsets", off.shape, "steps", int(off[-1]))
print("split code counts", dict(collections.Counter(split.tolist())))
# date from key csds/YYYY/MM/DD/uuid/csds
dates = [k.split("/")[1] + "-" + k.split("/")[2] + "-" + k.split("/")[3] for k in keys]
dc = collections.Counter(d[:7] for d in dates)
print("match counts by month", sorted(dc.items()))
print("n keys with unexpected format", sum(1 for k in keys if not k.startswith("csds/") or not k.endswith("/csds")))
# duplicate (match, round_num) pairs in the export
pairs = collections.Counter(zip(rm.tolist(), rn.tolist()))
print("duplicate (match, round_num) pairs", sum(1 for v in pairs.values() if v > 1))
nsteps = np.diff(off)
print("steps per round min", int(nsteps.min()), "max", int(nsteps.max()), "rounds with 0 steps", int((nsteps == 0).sum()))
print("rounds with freeze residue mod 64 nonzero (should be 0 by design of freeze)", int(((rf % 64) != 0).sum()), "freeze residues count", len(set((rf % 64).tolist())))
print("planted rounds", int((rp >= 0).sum()), "unplanted", int((rp < 0).sum()))
# unplanted rounds: steps should be <= 116 by serving cap
print("unplanted steps max", int(nsteps[rp < 0].max()), "planted steps max", int(nsteps[rp >= 0].max()))
print("rounds with steps > 116 and unplanted", int(((rp < 0) & (nsteps > 116)).sum()))
