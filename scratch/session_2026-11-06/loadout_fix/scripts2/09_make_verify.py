import json, sys, hashlib, random
import numpy as np
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
keys = json.load(open(S + "/match_keys.json"))
off = np.load(S + "/offsets.npy").astype(np.int64)
rm = np.load(S + "/round_match.npy").astype(np.int64)
key_m = {k: i for i, k in enumerate(keys)}
req = json.load(open(S + "/out/local_req_keys.json"))
rng = random.Random(20261105)
sel = sorted(rng.sample(req, 200))
order = np.argsort(rm, kind="stable")
rm_sorted = rm[order]
arr16 = np.load(S + "/out/loadout_step_i16.npy", mmap_mode="r")
arr_eq = np.load(S + "/out/loadout_equip_i32.npy", mmap_mode="r")
local = {}
for mk in sel:
    m = key_m[mk]
    lo = np.searchsorted(rm_sorted, m, side="left")
    hi = np.searchsorted(rm_sorted, m, side="right")
    rr = np.sort(order[lo:hi])
    S_all = np.concatenate([np.arange(off[r], off[r + 1]) for r in rr])
    a = np.ascontiguousarray(np.asarray(arr16[S_all]).astype(np.int16)).tobytes()
    b = np.ascontiguousarray(np.asarray(arr_eq[S_all]).astype(np.int32)).tobytes()
    local[mk] = hashlib.sha256(a + b).hexdigest()
json.dump(local, open(S + "/out/local_hashes_sel.json", "w"))
body = open(S + "/scripts2/09_verify_space_body.py", encoding="utf-8").read()
open(S + "/scripts2/09_verify_job.py", "w", encoding="utf-8").write(body.replace("__SEL__", repr(sel)))
print("selected", len(sel), "local hashes", len(local), "job written")
