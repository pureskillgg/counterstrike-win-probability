import glob
import json
import re
from collections import Counter

import numpy as np
import polars as pl

B = "/home/sagemaker-user/adam/builder"
D = B + "/final/export_clean_v2"
keys = json.load(open(D + "/match_keys.json"))
names = json.load(open(D + "/loadout_names.json"))
off = np.load(D + "/offsets.npy")
po = np.load(D + "/pos_offsets.npy")
rm = np.load(D + "/round_match.npy")
S = np.load(D + "/steps_alive.npy", mmap_mode="r")
L = np.load(D + "/loadout_round.npy", mmap_mode="r")

pk = set()
for f in glob.glob(B + "/final/pos_parts/bucket_c*.parquet"):
    pk.update(pl.read_parquet(f, columns=["match_key"])["match_key"].to_list())
grp = np.array([1 if k in pk else 2 for k in keys], dtype=np.int64)  # 1 = in pos_parts (bucket), 2 = not (laptop)
print("match groups: in pos_parts", int((grp == 1).sum()), "| not in pos_parts", int((grp == 2).sum()))

date_re = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")
dates = Counter()
for k in keys:
    m = date_re.search(k)
    dates[(m.group(1), m.group(2)) if m else ("?", "?")] += 1
print("export keys by year/month (from key path):", sorted(dates.items()))
oct_late = sum(1 for k in keys if re.search(r"/2026/10/(2[6-9]|3[01])/", k))
print("export keys dated 2026/10/26-31 (from key path):", oct_late)
print("sample key example:", keys[0])

# freeze-second agreement, by source group
rgrp = grp[rm]
first = off[:-1]
ca = np.asarray(L[:, names.index("CT_alive")]).astype(np.int64)
ta = np.asarray(L[:, names.index("T_alive")]).astype(np.int64)
s0c = np.asarray(S[first, 0]).astype(np.int64)
s0t = np.asarray(S[first, 1]).astype(np.int64)
for g in (1, 2):
    sel = rgrp == g
    print(f"group {g}: rounds {int(sel.sum())} | CT mismatch {int(((s0c != ca) & sel).sum())} | T mismatch {int(((s0t != ta) & sel).sum())}")
    ct_pairs = Counter(zip(ca[sel & (s0c != ca)].tolist(), s0c[sel & (s0c != ca)].tolist()))
    print(f"   group {g} CT mismatch pairs (loadout alive, steps alive) top:", ct_pairs.most_common(5))

# pos sentinel share by group
pi8 = np.load(D + "/pos_int8.npy", mmap_mode="r")
pu16 = np.load(D + "/pos_uint16.npy", mmap_mode="r")
pf = np.load(D + "/pos_float16.npy", mmap_mode="r")
cnt = np.diff(po)
starts = po[:-1]
checks = {
    "c_site_hi (int8 -1)": np.asarray(pi8[:, 0]) == -1,
    "c_blind_n (int8 -1)": np.asarray(pi8[:, 11]) == -1,
    "c_nearest_site_mean (uint16 65535)": np.asarray(pu16[:, 0]) == 65535,
    "defuse_slack (nan)": np.isnan(np.asarray(pf).astype(np.float32)),
}
for label, mask in checks.items():
    per_round = np.add.reduceat(mask.astype(np.int64), starts)
    for g in (1, 2):
        sel = rgrp == g
        steps_g = int(cnt[sel].sum())
        miss_g = int(per_round[sel].sum())
        print(f"{label:38s} group {g}: steps {steps_g} missing {miss_g} ({100.0 * miss_g / max(steps_g, 1):.4f}%)")
    print(f"{label:38s} all: missing {int(mask.sum())}")
