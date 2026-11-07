"""Count-only check: de_mirage matches per export split and per v1/v2 membership. Prints counts only, never keys."""
import json
import os

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
G = "G:/pureskill/csds-cs2/data/combined"
MAPS = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]

keys = json.load(open(os.path.join(D, "match_keys.json")))
ms = np.load(os.path.join(D, "match_split.npy"))
rm = np.load(os.path.join(D, "round_match.npy"))
rmap = np.load(os.path.join(D, "round_map.npy"))
print("n_keys", len(keys), "| match_split", ms.dtype, ms.shape, "| round_match", rm.dtype, rm.shape,
      "| round_map", rmap.dtype, rmap.shape)
print("match_split value counts", dict(zip(*[x.tolist() for x in np.unique(ms, return_counts=True)])))
print("round_map value counts", dict(zip(*[x.tolist() for x in np.unique(rmap, return_counts=True)])))

u, idx = np.unique(rm, return_index=True)
first_map = np.full(len(keys), -1, dtype=np.int64)
first_map[u] = rmap[idx]
print("matches with a round:", int(len(u)), "| matches with no round:", int(len(keys) - len(u)))
print("every round agrees with its match map:", bool(np.all(rmap.astype(np.int64) == first_map[rm])))

assign_v2 = json.load(open(os.path.join(G, "split_manifest_v2.json")))["assign"]
assign_v1 = json.load(open(os.path.join(G, "split_manifest_v1_archive.json")))["assign"]
print("v2 assign entries", len(assign_v2), "| v1 archive assign entries", len(assign_v1))
print("export keys found in v2 assign:", sum(k in assign_v2 for k in keys), "| in v1 archive assign:", sum(k in assign_v1 for k in keys))

# map code -> label, by majority of v2 labels for export keys
from collections import Counter
pair = Counter()
for i, k in enumerate(keys):
    if k in assign_v2:
        pair[(int(ms[i]), assign_v2[k])] += 1
print("split code vs v2 label pairs (code, label): count", dict(sorted(pair.items(), key=lambda t: (t[0][0], t[0][1]))))

label_of = {}
for code in np.unique(ms).tolist():
    cands = {lab: c for (cd, lab), c in pair.items() if cd == code}
    label_of[code] = max(cands, key=cands.get) if cands else None
print("code -> label used for table", label_of)

in_v1 = np.array([k in assign_v1 for k in keys])
print("export matches in v1 archive:", int(in_v1.sum()), "| appended (not in v1):", int((~in_v1).sum()))

rows = []
for grp, mask in (("all", np.ones(len(keys), bool)), ("v1_frozen", in_v1), ("v2_appended", ~in_v1)):
    for code in sorted(np.unique(ms).tolist()):
        lab = label_of[code]
        for mi, mn in enumerate(MAPS):
            n = int(np.sum(mask & (ms == code) & (first_map == mi)))
            rows.append((grp, lab, mn, n))
print("group | split label | map | matches")
for r in rows:
    if r[2] == "de_mirage" or r[0] == "all":
        print(" | ".join(str(x) for x in r))

tot = {}
for grp in ("all", "v1_frozen", "v2_appended"):
    for lab in ("train", "val", "test"):
        tot[(grp, lab, "de_mirage")] = sum(r[3] for r in rows if r[0] == grp and r[1] == lab and r[2] == "de_mirage")
print("de_mirage matches by group and label:", {f"{g}/{l}": v for (g, l, _), v in tot.items()})
print("de_mirage matches in any split (all):", sum(r[3] for r in rows if r[0] == "all" and r[2] == "de_mirage"))
print("per-split totals (all maps):", {lab: sum(r[3] for r in rows if r[0] == "all" and r[1] == lab) for lab in ("train", "val", "test")})
