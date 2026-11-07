"""c_exposed float64 fix, export side (runs on the space).

Re-assembles the 24 position features with the float64 cone sources (pos_parts_f64 bucket chunks + laptop_f64 day files), the same
join, precedence (bucket first, then laptop) and int8 rounding as pos_assemble.py. Then:
  - the four cone columns (c_sees_enemy, t_sees_enemy, c_exposed, t_exposed) in pos_int8 are patched in export_clean_v4 ONLY where the new
    value differs from export_clean_v2; every other cell is copied unchanged from v2;
  - the other int8 columns, uint16, uint8 and float16 features are compared with v2 (not written) as a regression check.
Writes: export_clean_v4 (copies of the v2 arrays, patched pos_int8) and fixes/cexp_f64/out/pos_v4_changes.csv (old and new per changed cell).
Never writes into export_clean_v2.
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl

t0 = time.time()
FIN = os.path.expanduser("~/adam/builder/final")
V2 = FIN + "/export_clean_v2"
V4 = FIN + "/export_clean_v4"
FX = os.path.expanduser("~/adam/builder/fixes/cexp_f64")
OUT = FX + "/out"
os.makedirs(OUT, exist_ok=True)
CONE = ["c_sees_enemy", "t_sees_enemy", "c_exposed", "t_exposed"]
INT8 = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_in_contact", "t_in_contact", "c_sees_enemy",
        "t_sees_enemy", "c_exposed", "t_exposed", "c_n_bomb_500", "c_blind_n", "t_blind_n"]
U16 = ["c_nearest_site_mean", "t_nearest_site_mean", "c_spread", "t_spread", "ct_t_centroid_dist",
       "min_enemy_dist", "c_min_bomb_dist", "t_min_bomb_dist"]
U8 = ["c_blind_sum", "t_blind_sum"]
F16 = ["defuse_slack"]
ALL = INT8 + U16 + U8 + F16
assert len(ALL) == 24 and len(set(ALL)) == 24
BOMB = ["c_n_bomb_500", "c_min_bomb_dist", "t_min_bomb_dist", "defuse_slack"]

keys = json.load(open(V2 + "/match_keys.json"))
mid_of = {k: i for i, k in enumerate(keys)}
off = np.load(V2 + "/offsets.npy")
rm = np.load(V2 + "/round_match.npy").astype(np.int64)
rn = np.load(V2 + "/round_num.npy").astype(np.int64)
frz = np.load(V2 + "/round_freeze.npy").astype(np.int64)
n = np.diff(off)
cnt = (n + 1) // 2
pos_off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
E, N = int(pos_off[-1]), len(rn)
print("rounds", N, "| even steps E", E, flush=True)
pos_off_v2 = np.load(V2 + "/pos_offsets.npy")
assert np.array_equal(pos_off_v2, pos_off), "pos_offsets differ from v2 step rule"

rid = np.repeat(np.arange(N, dtype=np.int64), cnt)
j = np.arange(E, dtype=np.int64) - pos_off[rid]
tick = frz[rid] + 64 * (2 * j)
T = pl.DataFrame({"tgt": np.arange(E, dtype=np.int64), "mid": rm[rid], "round": rn[rid], "tick": tick})

parts = sorted(glob.glob(FX + "/pos_parts_f64/bucket_c*.parquet"))
lap = sorted(glob.glob(FX + "/laptop_f64/*.parquet"))
print("bucket parts", len(parts), "| laptop f64 day files", len(lap), flush=True)
F = pl.concat([pl.read_parquet(f).select(["match_key", "round", "tick"] + ALL) for f in parts + lap], how="vertical_relaxed")
F = F.with_columns(pl.col("match_key").replace_strict(mid_of, default=None, return_dtype=pl.Int64).alias("mid"),
                   pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).drop_nulls(["mid"])
F = F.with_columns(pl.col("mid").cast(pl.Int64)).drop("match_key")
dups = F.group_by(["mid", "round", "tick"]).len().filter(pl.col("len") > 1).height
print("feature rows", F.height, "| duplicate (mid, round, tick)", dups, flush=True)
F = F.unique(["mid", "round", "tick"], keep="first", maintain_order=True)

chk = {c: {"mismatch_vs_v2": 0} for c in ALL}
pi8_v2 = np.load(V2 + "/pos_int8.npy", mmap_mode="r")
pu16_v2 = np.load(V2 + "/pos_uint16.npy", mmap_mode="r")
pu8_v2 = np.load(V2 + "/pos_uint8.npy", mmap_mode="r")
pf16_v2 = np.load(V2 + "/pos_float16.npy", mmap_mode="r")
changes = []
BLOCK = 6000
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
    a = {c: J[c].to_numpy().astype(np.float64) for c in ALL}
    # int8 features, same rounding and sentinel as pos_assemble.py
    v8 = {}
    for i, c in enumerate(INT8):
        v8[c] = np.where(np.isnan(a[c]), -1, np.clip(np.round(a[c]), 0, 5)).astype(np.int8)
        old = np.asarray(pi8_v2[t_lo:t_hi, i])
        d = v8[c] != old
        if d.any():
            chk[c]["mismatch_vs_v2"] += int(d.sum())
            if c in CONE:
                idx = np.nonzero(d)[0]
                for q in idx.tolist():
                    tg = t_lo + q
                    changes.append({"feature": c, "step_index": tg, "match_key": keys[int(rm[rid[tg]])], "round": int(rn[rid[tg]]),
                                    "tick": int(tick[tg]), "old": int(old[q]), "new": int(v8[c][q])})
    # other features: regression check against v2, not written
    for i, c in enumerate(U16):
        v = np.where(np.isnan(a[c]), 65535, np.clip(np.round(a[c]), 0, 65534)).astype(np.uint16)
        chk[c]["mismatch_vs_v2"] += int((v != np.asarray(pu16_v2[t_lo:t_hi, i])).sum())
    for i, c in enumerate(U8):
        v = np.where(np.isnan(a[c]), 255, np.clip(np.round(a[c] * 10), 0, 254)).astype(np.uint8)
        chk[c]["mismatch_vs_v2"] += int((v != np.asarray(pu8_v2[t_lo:t_hi, i])).sum())
    vf = a[F16[0]].astype(np.float16)
    of = np.asarray(pf16_v2[t_lo:t_hi])
    eqf = (vf == of) | (np.isnan(vf) & np.isnan(of))
    chk[F16[0]]["mismatch_vs_v2"] += int((~eqf).sum())
    print(f"  block {b0}-{b1}: steps {t_hi - t_lo} | secs {time.time() - t0:.0f}", flush=True)

for c in ALL:
    print(f"  {c}: mismatch vs v2 = {chk[c]['mismatch_vs_v2']}", flush=True)
ch = pl.DataFrame(changes) if changes else pl.DataFrame(schema={"feature": pl.String, "step_index": pl.Int64, "match_key": pl.String,
                                                                 "round": pl.Int64, "tick": pl.Int64, "old": pl.Int64, "new": pl.Int64})
ch.write_csv(OUT + "/pos_v4_changes.csv")
summary = {"cone_cells_changed": ch.height, "by_feature": {c: int(ch.filter(pl.col("feature") == c).height) for c in CONE},
           "regression_mismatch_vs_v2": {c: chk[c]["mismatch_vs_v2"] for c in ALL if c not in CONE}}

# write export_clean_v4: copy v2 (everything), then patch the cone columns only where they changed
os.makedirs(V4, exist_ok=True)
for fn in sorted(os.listdir(V2)):
    src, dst = os.path.join(V2, fn), os.path.join(V4, fn)
    if os.path.isfile(src) and not os.path.exists(dst):
        shutil.copyfile(src, dst)
pi8_v4 = np.load(V4 + "/pos_int8.npy", mmap_mode="r+")
for c in CONE:
    k = INT8.index(c)
    sub = ch.filter(pl.col("feature") == c)
    if sub.height:
        pi8_v4[sub["step_index"].to_numpy(), k] = sub["new"].to_numpy().astype(np.int8)
pi8_v4.flush()
del pi8_v4
summary["v4_pos_int8_cone_patched"] = True
summary["sec"] = round(time.time() - t0, 1)
json.dump(summary, open(OUT + "/pos_v4_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
print("done", f"{time.time() - t0:.0f}s", flush=True)
