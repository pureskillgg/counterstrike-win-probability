"""Trace the c_exposed mismatch key to its export source file. Read-only."""
import glob
import json

import numpy as np
import polars as pl

B = "/home/sagemaker-user/adam/builder"
FIN = B + "/final"
V2 = FIN + "/export_clean_v2"
KEY = "csds/2026/10/11/ad02c41a-b6e5-4631-9db0-a8b9ec8d6d7c/csds"
ROUND, TICK = 3, 17056

print("== pos_bucket.log lines for c050 / c060 / c000:")
for line in open(B + "/pos_bucket.log"):
    if line.startswith(("c050", "c060", "c000", "c001")):
        print(" ", line.rstrip())

print("\n== which pos_parts files contain the key:")
hits = []
for f in sorted(glob.glob(FIN + "/pos_parts/bucket_c*.parquet")):
    d = pl.scan_parquet(f).filter(pl.col("match_key") == KEY).select("match_key", "round", "tick", "c_exposed").collect()
    if d.height:
        hits.append((f, d.height))
print(" pos_parts hits:", hits)

print("\n== which laptop day files contain the key:")
lap = sorted(glob.glob(B + "/laptop_pos/every2_sites_v1/*.parquet"))
lhits = []
for f in lap:
    d = pl.scan_parquet(f).filter(pl.col("match_key") == KEY).select("match_key", "round", "tick", "c_exposed").collect()
    if d.height:
        lhits.append((f, d.height))
print(" laptop hits:", lhits)
for f, _ in lhits:
    d = pl.read_parquet(f).filter((pl.col("match_key") == KEY) & (pl.col("round") == ROUND) & (pl.col("tick") == TICK))
    print(" laptop row in", f, "->", d.select("round", "tick", "c_exposed", "c_site_hi", "c_in_contact", "c_sees_enemy", "t_sees_enemy").rows())
    print(" laptop schema c_exposed dtype:", d.schema.get("c_exposed"))

print("\n== export-era key list coverage: whether this key sits in laptop day files or in pos_parts")
mk = json.load(open(V2 + "/match_keys.json"))
mid = mk.index(KEY)
print(" export mid for key:", mid)
rm = np.load(V2 + "/round_match.npy").astype(np.int64)
rn = np.load(V2 + "/round_num.npy").astype(np.int64)
frz = np.load(V2 + "/round_freeze.npy").astype(np.int64)
pos_off = np.load(V2 + "/pos_offsets.npy").astype(np.int64)
pi8 = np.load(V2 + "/pos_int8.npy", mmap_mode="r")
idx = np.where((rm == mid) & (rn == ROUND))[0]
print(" export round index for (mid, round):", idx.tolist(), "freeze", frz[idx].tolist())
for r in idx:
    j = (TICK - int(frz[r])) // 128
    e = int(pos_off[r]) + j
    print(" export step e:", e, "j:", j, "c_exposed export:", int(np.asarray(pi8[e, 8])), "row:", np.asarray(pi8[e]).tolist())
