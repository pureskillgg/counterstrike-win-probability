import numpy as np, json, polars as pl, collections, os
E = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
F = "/home/sagemaker-user/adam/builder/final"
for f in ["loadout_round.npy","steps_alive.npy","offsets.npy","round_match.npy","round_num.npy","round_freeze.npy","pos_int8.npy","pos_uint16.npy","pos_uint8.npy","pos_float16.npy","pos_offsets.npy","match_split.npy","loadout_flag.npy"]:
    m = np.load(E+"/"+f, mmap_mode='r'); print(f, m.shape, m.dtype, "bytes", os.path.getsize(E+"/"+f))
L = np.load(E+"/loadout_round.npy", mmap_mode='r')
S = np.load(E+"/steps_alive.npy", mmap_mode='r')
O = np.load(E+"/offsets.npy")
RM = np.load(E+"/round_match.npy"); RN = np.load(E+"/round_num.npy")
PO = np.load(E+"/pos_offsets.npy")
print("offsets len", len(O), "O[0]", O[0], "O[-1]", O[-1], "steps rows", S.shape[0], "pos_offsets last", PO[-1], "pos_offsets len", len(PO))
names = json.load(open(E+"/loadout_names.json"))
print("CT_alive idx", names.index("CT_alive"), "T_alive idx", names.index("T_alive"), "T_kit_n idx", names.index("T_kit_n"), "CT_kit_n idx", names.index("CT_kit_n"))
keys = json.load(open(E+"/match_keys.json"))
cm = pl.read_parquet(F+"/clean_matches.parquet", columns=["match_key","src"])
src = dict(zip(cm["match_key"].to_list(), cm["src"].to_list()))
lo = O[:-1]; hi = O[1:]
valid = hi > lo
print("rounds", L.shape[0], "rounds with steps", int(valid.sum()), "rounds zero steps", int((~valid).sum()))
ct_loadout = np.asarray(L[:, names.index("CT_alive")])
t_loadout = np.asarray(L[:, names.index("T_alive")])
ct_grid = np.asarray(S[lo.clip(max=S.shape[0]-1), 0]); t_grid = np.asarray(S[lo.clip(max=S.shape[0]-1), 1])
mct = valid & (ct_loadout != ct_grid); mt = valid & (t_loadout != t_grid)
print("CT mismatch", int(mct.sum()), "T mismatch", int(mt.sum()), "any mismatch", int((mct|mt).sum()))
print("CT pairs (loadout,grid)", collections.Counter(zip(ct_loadout[mct].tolist(), ct_grid[mct].tolist())).most_common(6))
print("T pairs (loadout,grid)", collections.Counter(zip(t_loadout[mt].tolist(), t_grid[mt].tolist())).most_common(6))
print("CT_alive loadout counts", collections.Counter(ct_loadout.tolist()))
print("T_alive loadout counts", collections.Counter(t_loadout.tolist()))
# by source
rs = np.asarray(RM)
rsrc = np.array([src.get(keys[m], "NA") for m in rs[:]]) if False else None
srcs = collections.Counter(); mm = collections.Counter(); tot = collections.Counter()
mapping = {}
for i, m in enumerate(rs.tolist()):
    mapping[i] = src.get(keys[m], "NA")
for i in np.nonzero(valid)[0].tolist():
    s = mapping[i]; tot[s] += 1
    if mct[i] or mt[i]: mm[s] += 1
print("rounds by src (valid)", dict(tot), "rounds with any mismatch by src", dict(mm))
print("T_kit_n>0 rounds", int((np.asarray(L[:, names.index("T_kit_n")])>0).sum()), "max", float(np.asarray(L[:, names.index("T_kit_n")]).max()), "CT_kit_n>0", int((np.asarray(L[:, names.index("CT_kit_n")])>0).sum()))
# example rounds
for pref, rn in [("9d26354b3", 4), ("ba1358958", 7)]:
    idx = [i for i, m in enumerate(rs.tolist()) if keys[m].split("/")[-2].startswith(pref) and int(RN[i]) in (rn, rn-1, rn+1)]
    for i in idx:
        print("example", pref, "round_idx", i, "round_num", int(RN[i]), "match", keys[rs[i]][-40:], "loadout CT/T", int(ct_loadout[i]), int(t_loadout[i]), "grid k0 CT/T", int(ct_grid[i]), int(t_grid[i]), "steps", int(hi[i]-lo[i]), "src", mapping[i])
pi = np.load(E+"/pos_int8.npy", mmap_mode='r')
print("pos_int8 rows", pi.shape[0], "c_site_hi == -1", int((np.asarray(pi[:,0]) == -1).sum()))
