"""Three-way check on the 1,878 matches in final/pos_parity c050+c060.

  A) laptop day files (laptop_pos/every2_sites_v1) vs export pos_* arrays, encoded with the pos_assemble rule
  B) laptop day files vs final/pos_parity raw float32 values (exact equality, NaN == NaN)
Read-only. Matches are selected from the parity files only; no split labels or holdout lists are read.
"""
import glob
import json

import numpy as np
import polars as pl

FIN = "/home/sagemaker-user/adam/builder/final"
V2 = FIN + "/export_clean_v2"
LAP = "/home/sagemaker-user/adam/builder/laptop_pos/every2_sites_v1"
INT8 = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_in_contact", "t_in_contact", "c_sees_enemy",
        "t_sees_enemy", "c_exposed", "t_exposed", "c_n_bomb_500", "c_blind_n", "t_blind_n"]
U16 = ["c_nearest_site_mean", "t_nearest_site_mean", "c_spread", "t_spread", "ct_t_centroid_dist",
       "min_enemy_dist", "c_min_bomb_dist", "t_min_bomb_dist"]
U8 = ["c_blind_sum", "t_blind_sum"]
F16 = ["defuse_slack"]
ALL = INT8 + U16 + U8 + F16

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
rt = pl.DataFrame({"mid": rm, "round": rn, "freeze": frz, "ridx": np.arange(N, dtype=np.int64)})


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
    out[F16[0]] = np.asarray(pf16[e])
    return out


def encode(c, a):
    miss = np.isnan(a)
    if c in INT8:
        return np.where(miss, -1, np.clip(np.round(np.where(miss, 0.0, a)), 0, 5)).astype(np.int64)
    if c in U16:
        return np.where(miss, 65535, np.clip(np.round(np.where(miss, 0.0, a)), 0, 65534)).astype(np.int64)
    if c in U8:
        return np.where(miss, 255, np.clip(np.round(np.where(miss, 0.0, a) * 10), 0, 254)).astype(np.int64)
    return a.astype(np.float16)


def mismatch(c, exp, got):
    if c in F16:
        return ~((np.isnan(exp) & np.isnan(got)) | (exp == got))
    return exp != got


# parity keys (c050 + c060)
par = []
for c in ["c050", "c060"]:
    d = pl.read_parquet(f"{FIN}/pos_parity/bucket_{c}.parquet").with_columns(pl.lit(c).alias("bucket"))
    par.append(d)
par = pl.concat(par, how="vertical_relaxed")
pkeys = par["match_key"].unique().to_list()
print("parity rows", par.height, "| matches", len(pkeys), flush=True)

# laptop day files restricted to these matches
ldfs = []
for f in sorted(glob.glob(LAP + "/*.parquet")):
    d = pl.read_parquet(f)
    d = d.filter(pl.col("match_key").is_in(pkeys))
    if d.height:
        ldfs.append(d.with_columns(pl.lit(f.rsplit("/", 1)[1]).alias("src_file")))
lap = pl.concat(ldfs, how="vertical_relaxed")
print("laptop rows for these matches", lap.height, "| matches", lap["match_key"].n_unique(),
      "| source files", len(ldfs), flush=True)
lap_dup = lap.group_by(["match_key", "round", "tick"]).len().filter(pl.col("len") > 1).height
print("laptop duplicate (match_key, round, tick)", lap_dup, flush=True)
lap = lap.unique(["match_key", "round", "tick"], keep="first", maintain_order=True)
par_keys = set(zip(par["match_key"].to_list(), par["round"].to_list(), par["tick"].to_list()))
lap_keys = set(zip(lap["match_key"].to_list(), lap["round"].to_list(), lap["tick"].to_list()))
print("parity keys not in laptop", len(par_keys - lap_keys), "| laptop keys not in parity", len(lap_keys - par_keys), flush=True)

# B) laptop vs parity raw
pj = par.join(lap.select(["match_key", "round", "tick"] + ALL), on=["match_key", "round", "tick"], how="inner", suffix="_lap")
print("\n== B) laptop vs parity, raw float32 exact comparison, joined rows", pj.height, flush=True)
print(f"{'feature':24s} {'raw_diff':>9s} {'null_diff':>9s} verdict", flush=True)
for c in ALL:
    a = pj[c].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    b = pj[c + "_lap"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    raw = int(((a != b) & ~both_nan).sum())
    nd = int((np.isnan(a) != np.isnan(b)).sum())
    print(f"{c:24s} {raw:9d} {nd:9d} {'PASS' if raw == 0 and nd == 0 else 'FAIL'}", flush=True)
    if raw:
        idx = np.where((a != b) & ~both_nan)[0][:5]
        for i in idx:
            print(f"     example parity={a[i]!r} laptop={b[i]!r} key={pj['match_key'][int(i)]} r={pj['round'][int(i)]} t={pj['tick'][int(i)]}", flush=True)

# A) laptop -> export encoded
mk = lap.with_columns(
    pl.col("match_key").replace_strict(mid_of, default=None, return_dtype=pl.Int64).alias("mid"),
    pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
n_unmapped = int(mk["mid"].null_count())
mk = mk.drop_nulls("mid").join(rt, on=["mid", "round"], how="left")
n_no_round = int(mk["ridx"].null_count())
mk = mk.drop_nulls("ridx")
ridx = mk["ridx"].to_numpy().astype(np.int64)
sec = mk["tick"].to_numpy().astype(np.int64) - mk["freeze"].to_numpy().astype(np.int64)
on_grid = (sec >= 0) & (sec % 128 == 0)
jidx = np.where(on_grid, sec // 128, -1)
in_range = on_grid & (jidx < cnt[ridx])
print(f"\n== A) laptop -> export: unmapped {n_unmapped}, no export round {n_no_round}, off-grid {int((~on_grid).sum())}, "
      f"out-of-range {int((on_grid & ~in_range).sum())}, compared {int(in_range.sum())}", flush=True)
mk = mk.filter(pl.Series(in_range))
e = pos_off[ridx[in_range]] + jidx[in_range]
got_all = export_values(e)
print(f"{'feature':24s} {'n':>9s} {'EXACT_MM':>9s} verdict", flush=True)
for c in ALL:
    a = mk[c].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    exp = encode(c, a)
    mm = mismatch(c, exp, got_all[c])
    print(f"{c:24s} {len(a):9d} {int(mm.sum()):9d} {'PASS' if int(mm.sum()) == 0 else 'FAIL'}", flush=True)
    if mm.sum():
        idx = np.where(mm)[0][:5]
        for i in idx:
            print(f"     laptop={a[i]!r} expected={exp[i]!r} export={got_all[c][i]!r} key={mk['match_key'][int(i)]} r={mk['round'][int(i)]} t={mk['tick'][int(i)]} file={mk['src_file'][int(i)]}", flush=True)
print("\nDONE", flush=True)
