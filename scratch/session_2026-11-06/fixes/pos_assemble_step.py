"""Assemble the position features onto export_clean_v2's every-second steps (every export step), in mixed dtypes.

Steps: round r has n steps; the even steps are k = 0, 2, ... < n, i.e. ceil(n/2) per round, tick = freeze + 64 k.
Sources: final/pos_parts/bucket_c*.parquet (bucket matches) and laptop_pos/every2_sites_v1/*.parquet (Sep/Oct matches).
Output (final/export_clean_v2): pos_int8 (E,13), pos_uint16 (E,8), pos_uint8 (E,2), pos_float16 (E,), pos_offsets (N+1).
Sentinels: int8 -1, uint16 65535, uint8 255, float16 NaN. Units: distances in game units; blindness sums in 0.1 s.
"""
import glob
import json
import os
import time

import numpy as np
import polars as pl

FIN = os.path.expanduser("~/adam/builder/final")
V2 = FIN + "/export_clean_v2"  # input export (read only)
OUT = os.environ.get("POSA_OUT", FIN + "/export_clean_v4")  # new export folder; the v2 folder is never written
HOME = os.path.expanduser("~/adam/builder")
t0 = time.time()

INT8 = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_in_contact", "t_in_contact", "c_sees_enemy",
        "t_sees_enemy", "c_exposed", "t_exposed", "c_n_bomb_500", "c_blind_n", "t_blind_n"]
U16 = ["c_nearest_site_mean", "t_nearest_site_mean", "c_spread", "t_spread", "ct_t_centroid_dist",
       "min_enemy_dist", "c_min_bomb_dist", "t_min_bomb_dist"]
U8 = ["c_blind_sum", "t_blind_sum"]
F16 = ["defuse_slack"]
ALL = INT8 + U16 + U8 + F16
assert len(ALL) == 24 and len(set(ALL)) == 24

keys = json.load(open(V2 + "/match_keys.json"))
mid_of = {k: i for i, k in enumerate(keys)}
off = np.load(V2 + "/offsets.npy")
rm = np.load(V2 + "/round_match.npy").astype(np.int64)
rn = np.load(V2 + "/round_num.npy").astype(np.int64)
frz = np.load(V2 + "/round_freeze.npy").astype(np.int64)
n = np.diff(off)
cnt = n  # every export second: one pos row per export step, so pos_offsets == offsets
pos_off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
E, N = int(pos_off[-1]), len(rn)
assert E == int(off[-1]), "every-second layout must have one row per export step"
print("rounds", N, "| even steps E", E, flush=True)

rid = np.repeat(np.arange(N, dtype=np.int64), cnt)
j = np.arange(E, dtype=np.int64) - pos_off[rid]
tick = frz[rid] + 64 * j
T = pl.DataFrame({"tgt": np.arange(E, dtype=np.int64), "mid": rm[rid], "round": rn[rid], "tick": tick})

parts = sorted(glob.glob(FIN + "/pos_parts_every1/bucket_c*.parquet"))
lap = sorted(glob.glob(HOME + "/laptop_pos/every1_sites_v1/*.parquet"))
print("bucket parts", len(parts), "| laptop day files", len(lap), flush=True)
F = pl.concat([pl.read_parquet(f).select(["match_key", "round", "tick"] + ALL) for f in parts + lap], how="vertical_relaxed")
F = F.with_columns(pl.col("match_key").replace_strict(mid_of, default=None, return_dtype=pl.Int64).alias("mid"),
                   pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).drop_nulls(["mid"])
F = F.with_columns(pl.col("mid").cast(pl.Int64)).drop("match_key")
dups = F.group_by(["mid", "round", "tick"]).len().filter(pl.col("len") > 1).height
print("feature rows", F.height, "| duplicate (mid, round, tick)", dups, flush=True)
F = F.unique(["mid", "round", "tick"], keep="first", maintain_order=True)

pi8 = np.zeros((E, len(INT8)), dtype=np.int8)
pu16 = np.zeros((E, len(U16)), dtype=np.uint16)
pu8 = np.zeros((E, len(U8)), dtype=np.uint8)
pf16 = np.zeros(E, dtype=np.float16)
miss = {c: 0 for c in ALL}

BLOCK = 6000
for lo in range(0, N, 1)[::1]:
    pass
mids = rm
bounds = list(range(0, len(keys), BLOCK)) + [len(keys)]
for b0, b1 in zip(bounds[:-1], bounds[1:]):
    r0 = int(np.searchsorted(rm, b0, side="left"))
    r1 = int(np.searchsorted(rm, b1, side="left"))
    if r1 <= r0:
        continue
    t_lo, t_hi = int(pos_off[r0]), int(pos_off[r1])
    Tb = T.slice(t_lo, t_hi - t_lo)
    Fb = F.filter((pl.col("mid") >= b0) & (pl.col("mid") < b1))
    J = Tb.join(Fb, on=["mid", "round", "tick"], how="left").sort("tgt")
    assert J.height == t_hi - t_lo, "join changed the row count"
    loc = slice(0, t_hi - t_lo)
    for c in ALL:
        miss[c] += int(J[c].is_null().sum())
    a = {c: J[c].to_numpy().astype(np.float64) for c in ALL}
    for i, c in enumerate(INT8):
        v = np.where(np.isnan(a[c]), -1, np.clip(np.round(a[c]), 0, 5))
        pi8[t_lo:t_hi, i] = v.astype(np.int8)
    for i, c in enumerate(U16):
        v = np.where(np.isnan(a[c]), 65535, np.clip(np.round(a[c]), 0, 65534))
        pu16[t_lo:t_hi, i] = v.astype(np.uint16)
    for i, c in enumerate(U8):
        v = np.where(np.isnan(a[c]), 255, np.clip(np.round(a[c] * 10), 0, 254))
        pu8[t_lo:t_hi, i] = v.astype(np.uint8)
    pf16[t_lo:t_hi] = a[F16[0]].astype(np.float16)
    print(f"  block {b0}-{b1}: steps {t_hi - t_lo} | secs {time.time() - t0:.0f}", flush=True)

np.save(OUT + "/pos_int8.npy", pi8)
np.save(OUT + "/pos_uint16.npy", pu16)
np.save(OUT + "/pos_uint8.npy", pu8)
np.save(OUT + "/pos_float16.npy", pf16)
np.save(OUT + "/pos_offsets.npy", pos_off)
spec = {"steps": E, "rounds": N, "int8": INT8, "uint16": U16, "uint8_tenth_seconds": U8, "float16": F16,
        "sentinels": {"int8": -1, "uint16": 65535, "uint8": 255, "float16": "nan"},
        "step_rule": "every export step k=0..n-1 per round; tick = freeze + 64k; pos_offsets == offsets.npy",
        "missing_per_feature": miss,
        "bytes": int(pi8.nbytes + pu16.nbytes + pu8.nbytes + pf16.nbytes)}
json.dump(spec, open(OUT + "/pos_README.json", "w"), indent=1)
print(json.dumps({"missing_per_feature": miss, "MB": round(spec["bytes"] / 1e6, 1)}, indent=1), flush=True)
print("done", f"{time.time() - t0:.0f}s", flush=True)
