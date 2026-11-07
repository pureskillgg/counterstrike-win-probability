import json, os, csv, collections
import numpy as np
B2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2"
B3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout"
mk = json.load(open(os.path.join(B2, "match_keys.json")))
bk = json.load(open(os.path.join(B3, "built_match_keys.json")))
split = np.load(os.path.join(B2, "match_split.npy"))
round_match = np.load(os.path.join(B2, "round_match.npy")).astype(np.int64)
round_freeze = np.load(os.path.join(B2, "round_freeze.npy"))
offsets = np.load(os.path.join(B3, "offsets.npy"))
v2 = np.load(os.path.join(B2, "loadout_round.npy"), mmap_mode="r")
alive3 = np.load(os.path.join(B2, "steps_alive.npy"), mmap_mode="r")
step = np.load(os.path.join(B3, "loadout_step_i16.npy"), mmap_mode="r")
equip = np.load(os.path.join(B3, "loadout_equip_i32.npy"), mmap_mode="r")
with open(os.path.join(B3, "match_loadout_status.csv"), newline="") as fh:
    rdr = csv.reader(fh)
    hdr = next(rdr)
    print("status csv header:", hdr)
    status_rows = list(rdr)
print("status csv rows:", len(status_rows), "first:", status_rows[0])
idx_of = {k: i for i, k in enumerate(mk)}
bmask = np.zeros(len(mk), dtype=bool)
for k in bk:
    bmask[idx_of[k]] = True
cand = np.nonzero(bmask[round_match] & (split[round_match] != 2))[0]
s0 = offsets[cand].astype(np.int64)
R = len(cand)
st = np.asarray(step[s0]).astype(np.int64)
eq = np.asarray(equip[s0]).astype(np.int64)
al = np.asarray(alive3[s0]).astype(np.int64)
v2r = np.asarray(v2[cand]).astype(np.int64).reshape(R, 2, 22)
v3_fields = ["hp","armor_n","armor_sum","helmet_n","kit_n","defusing","ak","m4","awp","rifle","sniper_other","smg","heavy","no_primary","flash","smoke","he","fire","decoy"]
# non-alive comparison: 19 fields + equip, per side
nonalive_bad = np.zeros((R, 2), dtype=bool)
for side in (0, 1):
    for j in range(19):
        nonalive_bad[:, side] |= (st[:, side, j] != v2r[:, side, 1 + j])
    nonalive_bad[:, side] |= (eq[:, side] != v2r[:, side, 20])
alive_bad = (al != v2r[:, :, 0])
print("rounds R:", R)
print("rounds with any non-alive mismatch (either side):", int(nonalive_bad.any(axis=1).sum()))
print("side-rows with non-alive mismatch:", int(nonalive_bad.sum()), "of", 2 * R)
print("side-rows with alive mismatch (steps_alive vs v2 alive):", int(alive_bad.sum()), "of", 2 * R)
print("rounds with any alive mismatch:", int(alive_bad.any(axis=1).sum()))
d = (v2r[:, :, 0] - al)
print("alive diff (v2 - steps_alive) counts:", {int(k): int(v) for k, v in zip(*np.unique(d[alive_bad], return_counts=True))})
# which matches carry the non-alive mismatches
mids = round_match[cand]
bad_rounds = np.nonzero(nonalive_bad.any(axis=1))[0]
bad_matches = collections.Counter(int(mids[b]) for b in bad_rounds)
print("matches with >=1 non-alive mismatch round:", len(bad_matches), "| total mismatch rounds:", len(bad_rounds))
print("top 5 matches by mismatch rounds (match_key, count):")
for mi, c in bad_matches.most_common(5):
    print("  ", mk[mi], c)
status_by_key = {}
for r in status_rows:
    status_by_key[r[0]] = r
cnt_status = collections.Counter()
for mi in bad_matches:
    row = status_by_key.get(mk[mi])
    cnt_status[row[1] if row and len(row) > 1 else "no-status-row"] += 1
print("bad matches by status column 2:", dict(cnt_status))
# field breakdown for non-alive mismatches
fb = collections.Counter()
for side, pre in ((0, "CT"), (1, "T")):
    for j, f in enumerate(v3_fields):
        fb[f"{pre}_{f}"] = int((st[:, side, j] != v2r[:, side, 1 + j]).sum())
    fb[f"{pre}_equip"] = int((eq[:, side] != v2r[:, side, 20]).sum())
print("non-alive mismatch cells by field:", {k: v for k, v in fb.items() if v})
# examples of a non-alive mismatch
for b in bad_rounds[:4]:
    r = int(cand[b]); mi = int(mids[b])
    print("EXAMPLE round", r, "match", mk[mi], "split", int(split[mi]), "freeze", int(round_freeze[r]), "offset", int(s0[b]))
    for side, pre in ((0, "CT"), (1, "T")):
        diffs = []
        for j, f in enumerate(v3_fields):
            if st[b, side, j] != v2r[b, side, 1 + j]:
                diffs.append((f, int(st[b, side, j]), int(v2r[b, side, 1 + j])))
        if eq[b, side] != v2r[b, side, 20]:
            diffs.append(("equip", int(eq[b, side]), int(v2r[b, side, 20])))
        print("   ", pre, "alive(steps)", int(al[b, side]), "alive(v2)", int(v2r[b, side, 0]), "diffs (field, v3, v2):", diffs)
