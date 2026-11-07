import json
import numpy as np
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
FIN = B + "/final"
EXP = FIN + "/export_clean_v2"

mk = json.load(open(EXP + "/match_keys.json"))
idx = {k: i for i, k in enumerate(mk)}
rm = np.asarray(np.load(EXP + "/round_match.npy")).astype(np.int64)
rn = np.asarray(np.load(EXP + "/round_num.npy")).astype(np.int64)
rf = np.asarray(np.load(EXP + "/round_freeze.npy")).astype(np.int64)
po = np.asarray(np.load(EXP + "/pos_offsets.npy")).astype(np.int64)
cnt = np.diff(po)
pi8 = np.load(EXP + "/pos_int8.npy", mmap_mode="r")
M = 1 << 24


def comp(midx, rnd, tick):
    return (midx * 64 + rnd) * M + tick


def check(path, label):
    t = pq.read_table(path, columns=["match_key", "round", "tick", "c_site_hi"]).to_pydict()
    pk = np.array([idx.get(k, -1) for k in t["match_key"]], dtype=np.int64)
    pr = np.array(t["round"], dtype=np.int64)
    ptk = np.array(t["tick"], dtype=np.int64)
    pch = np.array([np.nan if v is None else float(v) for v in t["c_site_hi"]])
    keep = pk >= 0
    print(label, "rows", len(pk), "| rows whose match is in export", int(keep.sum()))
    mids = np.unique(pk[keep])
    R = np.nonzero(np.isin(rm, mids))[0]
    c = cnt[R]
    tot = int(c.sum())
    rep = np.repeat(R, c)
    offs = np.repeat(np.cumsum(c) - c, c)
    j = np.arange(tot) - offs
    row = po[rep] + j
    etick = rf[rep] + 128 * j
    ecomp = comp(rm[rep], rn[rep], etick)
    pcomp = comp(pk[keep], pr[keep], ptk[keep])
    print(" export rows for these matches", tot, "| export composite unique", len(np.unique(ecomp)) == len(ecomp),
          "| parity composite unique", len(np.unique(pcomp)) == len(pcomp))
    common, ei, pi = np.intersect1d(ecomp, pcomp, return_indices=True)
    print(" matched on (match, round, tick)", len(common),
          "| parity rows not in export", len(pcomp) - len(common),
          "| export rows not in parity", tot - len(common))
    ev = np.asarray(pi8[row[ei], 0]).astype(float)
    ev[ev == -1] = np.nan
    pv = pch[keep][pi]
    both_nan = np.isnan(ev) & np.isnan(pv)
    eq = both_nan | (ev == pv)
    print(" c_site_hi equal on matched rows", int(eq.sum()), "of", len(eq))
    bad = np.nonzero(~eq)[0][:5]
    print(" c_site_hi sample mismatches (export, parity)", [(float(ev[b]), float(pv[b])) for b in bad])


check(FIN + "/pos_parts/bucket_c000.parquet", "CALIBRATION pos_parts c000")
check(FIN + "/pos_parity/bucket_c050.parquet", "PARITY c050")
check(FIN + "/pos_parity/bucket_c060.parquet", "PARITY c060")
