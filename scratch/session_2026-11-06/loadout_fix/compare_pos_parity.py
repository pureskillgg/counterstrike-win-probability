"""Compare final/pos_parity bucket_c050 and bucket_c060 against export_clean_v2 pos_* arrays.

Encoding rule replicated exactly from pos_assemble.py (the encoder that wrote the export):
  int8   : NaN -> -1, else clip(round(x), 0, 5)
  uint16 : NaN -> 65535, else clip(round(x), 0, 65534)
  uint8  : NaN -> 255, else clip(round(10 x), 0, 254)     (tenth-seconds)
  float16: astype(float16), NaN kept
Export step for a parity row: round index ridx from (mid, round_num), j = (tick - freeze) / 128 (even second),
e = pos_offsets[ridx] + j.  Read-only; writes nothing on the space.
"""
import json

import numpy as np
import polars as pl

FIN = "/home/sagemaker-user/adam/builder/final"
V2 = FIN + "/export_clean_v2"
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
off = np.load(V2 + "/offsets.npy").astype(np.int64)
rm = np.load(V2 + "/round_match.npy").astype(np.int64)
rn = np.load(V2 + "/round_num.npy").astype(np.int64)
frz = np.load(V2 + "/round_freeze.npy").astype(np.int64)
pos_off = np.load(V2 + "/pos_offsets.npy").astype(np.int64)
cnt = (np.diff(off) + 1) // 2
N = len(rm)
pi8 = np.load(V2 + "/pos_int8.npy", mmap_mode="r")
pu16 = np.load(V2 + "/pos_uint16.npy", mmap_mode="r")
pu8 = np.load(V2 + "/pos_uint8.npy", mmap_mode="r")
pf16 = np.load(V2 + "/pos_float16.npy", mmap_mode="r")
E = pi8.shape[0]
print("export E", E, "| pos_offsets[-1]", int(pos_off[-1]), "| offsets rule holds", bool(np.array_equal(np.diff(pos_off), cnt)), flush=True)

rt = pl.DataFrame({"mid": rm, "round": rn, "freeze": frz, "ridx": np.arange(N, dtype=np.int64)})
print("export (mid, round) duplicate pairs", rt.group_by(["mid", "round"]).len().filter(pl.col("len") > 1).height, flush=True)


def export_values(e):
    out = {}
    a8 = np.asarray(pi8[e])
    for i, c in enumerate(INT8):
        out[c] = a8[:, i].astype(np.int64)
    a16 = np.asarray(pu16[e])
    for i, c in enumerate(U16):
        out[c] = a16[:, i].astype(np.int64)
    au8 = np.asarray(pu8[e])
    for i, c in enumerate(U8):
        out[c] = au8[:, i].astype(np.int64)
    out[F16[0]] = np.asarray(pf16[e])  # float16
    return out


def encode(c, a):
    miss = np.isnan(a)
    if c in INT8:
        r = np.round(np.where(miss, 0.0, a))
        clipped = (~miss) & ((r < 0) | (r > 5))
        return np.where(miss, -1, np.clip(r, 0, 5)).astype(np.int64), clipped
    if c in U16:
        r = np.round(np.where(miss, 0.0, a))
        clipped = (~miss) & ((r < 0) | (r > 65534))
        return np.where(miss, 65535, np.clip(r, 0, 65534)).astype(np.int64), clipped
    if c in U8:
        r = np.round(np.where(miss, 0.0, a) * 10)
        clipped = (~miss) & ((r < 0) | (r > 254))
        return np.where(miss, 255, np.clip(r, 0, 254)).astype(np.int64), clipped
    return a.astype(np.float16), np.zeros(len(a), dtype=bool)


def sentinel_mask(c, got):
    if c in INT8:
        return got == -1
    if c in U16:
        return got == 65535
    if c in U8:
        return got == 255
    return np.isnan(got)


def decode(c, got):
    if c in U8:
        return got.astype(np.float64) / 10.0
    return got.astype(np.float64)


def mismatch(c, exp, got):
    if c in F16:
        return ~((np.isnan(exp) & np.isnan(got)) | (exp == got))
    return exp != got


def process(c):
    df = pl.read_parquet(f"{FIN}/pos_parity/bucket_{c}.parquet")
    print(f"\n== bucket_{c}: rows {df.height}, matches {df['match_key'].n_unique()}, cols {df.width}", flush=True)
    df = df.with_columns(
        pl.col("match_key").replace_strict(mid_of, default=None, return_dtype=pl.Int64).alias("mid"),
        pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    n_unmapped = int(df["mid"].null_count())
    df = df.drop_nulls("mid")
    j = df.join(rt, on=["mid", "round"], how="left")
    n_no_round = int(j["ridx"].null_count())
    j = j.drop_nulls("ridx")
    ridx = j["ridx"].to_numpy().astype(np.int64)
    sec = j["tick"].to_numpy().astype(np.int64) - j["freeze"].to_numpy().astype(np.int64)
    on_grid = (sec >= 0) & (sec % 128 == 0)
    jidx = np.where(on_grid, sec // 128, -1)
    in_range = on_grid & (jidx < cnt[ridx])
    n_offgrid = int((~on_grid).sum())
    n_outrange = int((on_grid & ~in_range).sum())
    keep = in_range
    ridx_k = ridx[keep]
    e = pos_off[ridx_k] + jidx[keep]
    assert int(e.max()) < E and int(e.min()) >= 0
    print(f" unmapped match_key {n_unmapped} | parity row with no export round {n_no_round} | "
          f"tick off even-second grid {n_offgrid} | step index out of range {n_outrange} | compared rows {len(e)}", flush=True)
    dup_e = len(e) - len(np.unique(e))
    print(f" duplicate export steps within this file {dup_e}", flush=True)
    got_all = export_values(e)
    stats = {}
    for c2 in ALL:
        a = j[c2].cast(pl.Float64).fill_null(float("nan")).to_numpy()[keep]
        exp, clipped = encode(c2, a)
        got = got_all[c2]
        mm = mismatch(c2, exp, got)
        pm = np.isnan(a)
        sm = sentinel_mask(c2, got)
        ok = (~pm) & (~clipped)
        if ok.any():
            derr = float(np.max(np.abs(decode(c2, got)[ok] - a[ok])))
        else:
            derr = 0.0
        stats[c2] = {
            "n": int(len(a)), "parity_missing": int(pm.sum()), "export_sentinel": int(sm.sum()),
            "exact_mismatch": int(mm.sum()), "missing_vs_nonsentinel": int((pm & ~sm).sum()),
            "value_vs_sentinel": int((~pm & sm).sum()), "clipped": int(clipped.sum()),
            "max_decode_err": derr,
            "min": float(np.nanmin(a)) if np.isfinite(a).any() else float("nan"),
            "max": float(np.nanmax(a)) if np.isfinite(a).any() else float("nan"),
        }
    return stats, e, dict(n_unmapped=n_unmapped, n_no_round=n_no_round, n_offgrid=n_offgrid,
                          n_outrange=n_outrange, compared=len(e), dup_e=dup_e)


def print_table(title, stats):
    print(f"\n### {title}", flush=True)
    print(f"{'feature':24s} {'n':>9s} {'par_miss':>9s} {'exp_sent':>9s} {'EXACT_MM':>9s} {'miss<>sent':>10s} {'clipped':>8s} {'max_dec_err':>11s} {'min':>10s} {'max':>10s} verdict", flush=True)
    for c in ALL:
        s = stats[c]
        v = "PASS" if s["exact_mismatch"] == 0 else "FAIL"
        print(f"{c:24s} {s['n']:9d} {s['parity_missing']:9d} {s['export_sentinel']:9d} {s['exact_mismatch']:9d} "
              f"{s['missing_vs_nonsentinel'] + s['value_vs_sentinel']:10d} {s['clipped']:8d} {s['max_decode_err']:11.4g} "
              f"{s['min']:10.4g} {s['max']:10.4g} {v}", flush=True)


res = {}
all_e = []
info = {}
for c in ["c050", "c060"]:
    stats, e, inf = process(c)
    res[c] = stats
    info[c] = inf
    all_e.append(e)
    print_table(f"bucket_{c}", stats)

comb = {}
for c2 in ALL:
    comb[c2] = {}
    for k in ["n", "parity_missing", "export_sentinel", "exact_mismatch", "missing_vs_nonsentinel", "value_vs_sentinel", "clipped"]:
        comb[c2][k] = res["c050"][c2][k] + res["c060"][c2][k]
    comb[c2]["max_decode_err"] = max(res["c050"][c2]["max_decode_err"], res["c060"][c2]["max_decode_err"])
    comb[c2]["min"] = float(np.nanmin([res["c050"][c2]["min"], res["c060"][c2]["min"]]))
    comb[c2]["max"] = float(np.nanmax([res["c050"][c2]["max"], res["c060"][c2]["max"]]))
print_table("COMBINED c050+c060", comb)

E_all = np.concatenate(all_e)
print(f"\ncross-file duplicate export steps {len(E_all) - len(np.unique(E_all))}", flush=True)
mids = set()
for c in ["c050", "c060"]:
    dfk = pl.read_parquet(f"{FIN}/pos_parity/bucket_{c}.parquet", columns=["match_key"])
    mids |= set(mid_of[k] for k in dfk["match_key"].unique().to_list() if k in mid_of)
mids_arr = np.array(sorted(mids), dtype=np.int64)
rounds_t = np.where(np.isin(rm, mids_arr))[0]
ct = cnt[rounds_t]
starts = pos_off[rounds_t]
steps = np.repeat(starts, ct) + (np.arange(int(ct.sum())) - np.repeat(np.cumsum(ct) - ct, ct))
covered = np.zeros(E, dtype=bool)
covered[E_all] = True
unc = steps[~covered[steps]]
print(f"matches touched {len(mids_arr)} | rounds {len(rounds_t)} | export even-steps for these matches {len(steps)} | "
      f"covered by parity rows {int(covered[steps].sum())} | uncovered {len(unc)}", flush=True)
if len(unc):
    gu = export_values(unc)
    for c2 in ALL:
        sm = sentinel_mask(c2, gu[c2])
        print(f"  uncovered step, export non-sentinel for {c2:22s}: {int((~sm).sum())}", flush=True)
print("\nDONE", flush=True)
