"""Leakage check on the assembled position features (read-only).

1) At the freeze tick (k=0) and k=2, a feature that depends on the outcome would separate CT-win from T-win rounds.
   Blindness and bomb features cannot know the outcome yet at the start of the round, so their label separation should be ~0.
2) Plant events: count rounds with more than one bomb_begin_plant (the 'last plant' position used by the features).
3) Blindness at k=0 should be zero: no flash can have gone off before the freeze tick.
"""
import json
import os

import numpy as np
import polars as pl

FIN = os.path.expanduser("~/adam/builder/final")
V2 = FIN + "/export_clean_v2"
spec = json.load(open(V2 + "/pos_README.json"))
pi8 = np.load(V2 + "/pos_int8.npy", mmap_mode="r")
pu16 = np.load(V2 + "/pos_uint16.npy", mmap_mode="r")
pu8 = np.load(V2 + "/pos_uint8.npy", mmap_mode="r")
pf16 = np.load(V2 + "/pos_float16.npy", mmap_mode="r")
po = np.load(V2 + "/pos_offsets.npy")
lab = np.load(V2 + "/round_label.npy").astype(np.int64)
N = len(lab)
cnt = np.diff(po)


def decode(name, idx):
    if name in spec["int8"]:
        v = np.asarray(pi8[idx, spec["int8"].index(name)]).astype(np.float64)
        v[v < 0] = np.nan
    elif name in spec["uint16"]:
        v = np.asarray(pu16[idx, spec["uint16"].index(name)]).astype(np.float64)
        v[v == 65535] = np.nan
    elif name in spec["uint8_tenth_seconds"]:
        v = np.asarray(pu8[idx, spec["uint8_tenth_seconds"].index(name)]).astype(np.float64) / 10.0
        v[np.asarray(pu8[idx, spec["uint8_tenth_seconds"].index(name)]) == 255] = np.nan
    else:
        v = np.asarray(pf16[idx]).astype(np.float64)
    return v


names = spec["int8"] + spec["uint16"] + spec["uint8_tenth_seconds"] + spec["float16"]
print("feature                    k=0: mean|CT-win  mean|T-win  std-diff  missing   |  k=2: std-diff  missing", flush=True)
for name in names:
    out = []
    for j in (0, 1):
        ok = cnt >= j + 1
        r = np.where(ok)[0]
        idx = po[r] + j
        v = decode(name, idx)
        y = lab[r]
        m1 = np.nanmean(v[y == 1]) if np.isfinite(v[y == 1]).any() else np.nan
        m0 = np.nanmean(v[y == 0]) if np.isfinite(v[y == 0]).any() else np.nan
        sd = np.nanstd(v)
        d = (m1 - m0) / sd if sd > 0 else 0.0
        miss = float(np.isnan(v).mean())
        out.append((m1, m0, d, miss))
    flag = "  <-- CHECK" if abs(out[0][2]) > 0.05 and name.startswith(("c_blind", "t_blind", "c_min_bomb", "t_min_bomb", "defuse", "c_n_bomb")) else ""
    print(f"{name:26s} {out[0][0]:10.3f} {out[0][1]:10.3f} {out[0][2]:9.4f} {out[0][3]:8.4f}   | {out[1][2]:9.4f} {out[1][3]:8.4f}{flag}", flush=True)

blind_nonzero_k0 = {n: int(np.nansum(decode(n, po[:-1]) > 0)) for n in ["c_blind_n", "t_blind_n", "c_blind_sum", "t_blind_sum"]}
print("rounds with blindness > 0 at the freeze tick (k=0):", blind_nonzero_k0, flush=True)

plants = pl.read_parquet(FIN + "/plants_all.parquet", columns=["match_key", "round", "tick"])
multi = plants.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
print("rounds with more than one bomb_begin_plant (all export matches):", multi, "of", plants.select("match_key", "round").unique().height, flush=True)
