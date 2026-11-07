"""Reads the two named c_exposed rows from export_clean_v2 and export_clean_v4 pos_int8 (read-only) and the parity file value."""
import json
import numpy as np
import polars as pl

V2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
keys = json.load(open(V2 + "match_keys.json"))
kid = {k: i for i, k in enumerate(keys)}
rm = np.load(V2 + "round_match.npy").astype(np.int64)
rn = np.load(V2 + "round_num.npy").astype(np.int64)
rf = np.load(V2 + "round_freeze.npy").astype(np.int64)
pos_off = np.load(V2 + "pos_offsets.npy").astype(np.int64)
p2 = np.load(V2 + "pos_int8.npy", mmap_mode="r")
p4 = np.load(V4 + "pos_int8.npy", mmap_mode="r")
INT8 = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_in_contact", "t_in_contact", "c_sees_enemy",
        "t_sees_enemy", "c_exposed", "t_exposed", "c_n_bomb_500", "c_blind_n", "t_blind_n"]
par = pl.read_parquet("/home/sagemaker-user/adam/builder/final/pos_parity/bucket_c050.parquet")
named = [("csds/2026/10/11/ad02c41a-b6e5-4631-9db0-a8b9ec8d6d7c/csds", 3, 17056),
         ("csds/2026/10/11/9c525f5e-d246-4d3d-9539-efc1de2d4028/csds", 7, 49470)]
out = []
for mk, rd, tk in named:
    m = kid[mk]
    idx = np.nonzero((rm == m) & (rn == rd))[0]
    assert idx.size == 1, (mk, rd, idx.size)
    r = int(idx[0])
    k = (tk - int(rf[r])) // 64
    assert (tk - int(rf[r])) % 64 == 0 and k % 2 == 0, ("not an even export step", k)
    s = int(pos_off[r] + k // 2)
    rec = {"match_key": mk, "round": rd, "tick": tk, "step_index": s,
           "v2_c_exposed": int(p2[s, INT8.index("c_exposed")]), "v4_c_exposed": int(p4[s, INT8.index("c_exposed")]),
           "v2_c_sees_enemy": int(p2[s, INT8.index("c_sees_enemy")]), "v4_c_sees_enemy": int(p4[s, INT8.index("c_sees_enemy")]),
           "v2_t_exposed": int(p2[s, INT8.index("t_exposed")]), "v4_t_exposed": int(p4[s, INT8.index("t_exposed")]),
           "v2_t_sees_enemy": int(p2[s, INT8.index("t_sees_enemy")]), "v4_t_sees_enemy": int(p4[s, INT8.index("t_sees_enemy")])}
    pr = par.filter((pl.col("match_key") == mk) & (pl.col("round") == rd) & (pl.col("tick") == tk)).select("c_exposed")
    rec["parity_bucket_c050_c_exposed"] = float(pr["c_exposed"][0]) if pr.height else None
    out.append(rec)
print(json.dumps(out, indent=1), flush=True)
