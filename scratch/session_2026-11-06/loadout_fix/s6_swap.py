import json, os, csv, collections
import numpy as np
B2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
B3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout"
mk = json.load(open(os.path.join(B2, "match_keys.json")))
bk = json.load(open(os.path.join(B3, "built_match_keys.json")))
split = np.load(os.path.join(B2, "match_split.npy"))
round_match = np.load(os.path.join(B2, "round_match.npy")).astype(np.int64)
offsets = np.load(os.path.join(B3, "offsets.npy"))
v2 = np.load(os.path.join(B2, "loadout_round.npy"), mmap_mode="r")
step = np.load(os.path.join(B3, "loadout_step_i16.npy"), mmap_mode="r")
equip = np.load(os.path.join(B3, "loadout_equip_i32.npy"), mmap_mode="r")
rows = list(csv.reader(open(os.path.join(B3, "match_loadout_status.csv"), newline="")))
hdr = rows[0]; body = rows[1:]
ci = {h: i for i, h in enumerate(hdr)}
status = {r[ci["match_key"]]: r for r in body}
idx_of = {k: i for i, k in enumerate(mk)}
bmask = np.zeros(len(mk), dtype=bool)
for k in bk:
    bmask[idx_of[k]] = True
cand = np.nonzero(bmask[round_match] & (split[round_match] != 2))[0]
s0 = offsets[cand].astype(np.int64)
R = len(cand)
st = np.asarray(step[s0]).astype(np.int64)
eq = np.asarray(equip[s0]).astype(np.int64)
v2r = np.asarray(v2[cand]).astype(np.int64).reshape(R, 2, 22)
v3_all = np.concatenate([st, eq[:, :, None]], axis=2)            # (R,2,20): 19 fields + equip
v2_all = np.concatenate([v2r[:, :, 1:20], v2r[:, :, 20:21]], axis=2)  # (R,2,20)
direct = (v3_all == v2_all).all(axis=2)                           # (R,2)
swapped = (v3_all[:, ::-1, :] == v2_all).all(axis=2)              # v3 CT vs v2 T and v3 T vs v2 CT
bad = ~direct.all(axis=1)
print("rounds", R, "| rounds with a direct side mismatch (19 fields + equip):", int(bad.sum()))
print("of those, rounds where the sides match exactly when swapped:", int((bad & swapped.all(axis=1)).sum()))
print("of those, rounds where neither direct nor swapped matches on both sides:", int((bad & ~swapped.all(axis=1)).sum()))
print("side-rows in bad rounds that match swapped:", int((~direct & swapped)[bad].sum()), "of", int((~direct)[bad].sum()), "side-rows mismatched directly")
mids = round_match[cand]
badr = np.nonzero(bad)[0]
mc = collections.Counter(int(mids[b]) for b in badr)
print("matches with side mismatch:", len(mc))
for mi, c in mc.most_common(30):
    r = status.get(mk[mi])
    print("  match", mk[mi], "bad rounds", c, "| split", int(split[mi]), "| date", r[ci["date"]] if r else "?", "| src", r[ci["src"]] if r else "?", "| build_status", r[ci["build_status"]] if r else "?", "| rounds_built", r[ci["export_rounds"]] if r else "?")
# how many rounds in those matches are fine
for mi, c in mc.most_common(3):
    all_rounds = np.nonzero(mids == mi)[0]
    print("match", mk[mi], "built rounds", len(all_rounds), "bad", c, "good", int(len(all_rounds) - c))
print("mismatch-rounds source breakdown (src):", dict(collections.Counter((status[mk[mi]][ci["src"]] if mk[mi] in status else "?") for mi in mc)))
print("mismatch-rounds build_status breakdown:", dict(collections.Counter((status[mk[mi]][ci["build_status"]] if mk[mi] in status else "?") for mi in mc)))
print("total built matches by build_status:", dict(collections.Counter(r[ci["build_status"]] for r in body if r[ci["build_status"]] and int(r[ci["m"]]) >= 0 and mk and r[ci["match_key"]] in idx_of and bmask[idx_of[r[ci["match_key"]]]])))
