"""Rebuild export_clean_v2 on the space from export_clean (v1), with the split taken from the frozen manifest v2.

Drops the rounds with duplicated player ids (loadout_flag == 1), removes matches left with no rounds, renumbers the matches,
and writes match_split from split_manifest_v2 (train 0, val 1, test 2). Writes only to final/export_clean_v2 (new folder).
"""
import json
import os
import shutil

import numpy as np

FIN = os.path.expanduser("~/adam/builder/final")
E = FIN + "/export_clean"
OUT = FIN + "/export_clean_v2"
os.makedirs(OUT, exist_ok=True)

off = np.load(E + "/offsets.npy")
steps = np.load(E + "/steps_alive.npy", mmap_mode="r")
flag = np.load(E + "/loadout_flag.npy")
rm = np.load(E + "/round_match.npy")
keys = json.load(open(E + "/match_keys.json"))
keep = flag == 0
n = np.diff(off)

new_steps = np.asarray(steps)[np.repeat(keep, n)]
new_off = np.concatenate([[0], np.cumsum(n[keep])]).astype(np.int64)
for name in ["round_freeze", "round_plant", "round_label", "round_num", "round_map"]:
    np.save(OUT + "/" + name + ".npy", np.load(E + "/" + name + ".npy")[keep])
np.save(OUT + "/steps_alive.npy", new_steps)
np.save(OUT + "/offsets.npy", new_off)
np.save(OUT + "/loadout_round.npy", np.asarray(np.load(E + "/loadout_round.npy", mmap_mode="r"))[keep])
np.save(OUT + "/loadout_flag.npy", flag[keep])
for name in ["loadout_names.json", "maps.json"]:
    shutil.copy(E + "/" + name, OUT + "/" + name)

used = np.unique(rm[keep])
newid = -np.ones(len(keys), dtype=np.int64)
newid[used] = np.arange(len(used))
rm2 = newid[rm[keep]].astype(np.int32)
np.save(OUT + "/round_match.npy", rm2)
keys2 = [keys[i] for i in used]
json.dump(keys2, open(OUT + "/match_keys.json", "w"))

assign = json.load(open(FIN + "/split_manifest_v2.json", encoding="utf-8"))["assign"]
code = {"train": 0, "val": 1, "test": 2}
split = np.array([code[assign[k]] for k in keys2], dtype=np.uint8)
np.save(OUT + "/match_split.npy", split)

names = ["train", "val", "test"]
rep = {
    "matches": len(keys2), "rounds": int(keep.sum()), "steps": int(new_steps.shape[0]),
    "dropped_rounds_duplicate_player_id": int((~keep).sum()),
    "matches_removed_empty": int(len(keys) - len(keys2)),
    "matches_per_split": {names[s]: int((split == s).sum()) for s in range(3)},
    "rounds_per_split": {names[s]: int((split[rm2] == s).sum()) for s in range(3)},
    "offsets_ok": bool(new_off[-1] == new_steps.shape[0]),
    "all_keys_in_manifest": all(k in assign for k in keys2),
    "source": "export_clean (v1)", "split": "split_manifest_v2 (frozen v1 + appended 80/10/10 by map, seed 20261104)",
}
json.dump(rep, open(OUT + "/README.json", "w"), indent=1)
print(json.dumps(rep, indent=1), flush=True)
