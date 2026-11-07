import json, numpy as np
S = r"C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix/"
keys = json.load(open(S + "match_keys.json"))
rf = np.load(S + "round_freeze.npy"); rm = np.load(S + "round_match.npy"); rn = np.load(S + "round_num.npy"); off = np.load(S + "offsets.npy")
for name, a in [("round_freeze", rf), ("round_match", rm), ("round_num", rn), ("offsets", off)]:
    print(name, "dtype", a.dtype, "shape", a.shape, "min", a.min(), "max", a.max(), "first10", a[:10].tolist())
print("round_match unique", len(np.unique(rm)), "max idx", rm.max(), "keys len", len(keys))
print("round_num value counts head", np.unique(rn, return_counts=True)[0][:10].tolist(), np.unique(rn, return_counts=True)[1][:10].tolist())
# freeze mod 64 check
print("freeze mod 64 counts (top 5):", np.unique(rf % 64, return_counts=True)[0][:5].tolist(), np.unique(rf % 64, return_counts=True)[1][:5].tolist())
print("freeze mod 64 distinct:", len(np.unique(rf % 64)))
# rounds per match
rpm = np.bincount(rm, minlength=len(keys))
print("rounds per match min/max/mean", rpm.min(), rpm.max(), round(rpm.mean(), 2), "matches with 0 rounds", int((rpm == 0).sum()))
# first round of first match
m0 = np.where(rm == 0)[0]
print("match0 round rows", m0[:30].tolist(), "round_num", rn[m0][:30].tolist(), "freeze", rf[m0][:30].tolist())
print("keys[0] rounds", len(m0))
