
# ---- verification on the space: per-match hashes (sample), parity rows vs written arrays, ranges, row counts ----
import json, sys, time, hashlib
import numpy as np
import polars as pl

T0 = time.time()


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, file=sys.stderr, flush=True)


EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
STAGE = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout_stage/"
OUTD = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
keys = json.load(open(EXP + "match_keys.json"))
off = np.load(EXP + "offsets.npy").astype(np.int64)
rm = np.load(EXP + "round_match.npy").astype(np.int64)
NS = int(off[-1])
key_m = {k: i for i, k in enumerate(keys)}
SEL_KEYS = __SEL__
arr16 = np.load(OUTD + "loadout_step_i16.npy", mmap_mode="r")
arr_eq = np.load(OUTD + "loadout_equip_i32.npy", mmap_mode="r")
assert arr16.shape == (NS, 2, 19) and arr_eq.shape == (NS, 2)
res = {"hashes": {}}
rounds_by_m = {}
order = np.argsort(rm, kind="stable")
rm_sorted = rm[order]
for mk in SEL_KEYS:
    m = key_m[mk]
    lo = np.searchsorted(rm_sorted, m, side="left")
    hi = np.searchsorted(rm_sorted, m, side="right")
    rr = np.sort(order[lo:hi])
    S_all = np.concatenate([np.arange(off[r], off[r + 1]) for r in rr])
    a = np.ascontiguousarray(np.asarray(arr16[S_all]).astype(np.int16)).tobytes()
    b = np.ascontiguousarray(np.asarray(arr_eq[S_all]).astype(np.int32)).tobytes()
    res["hashes"][mk] = hashlib.sha256(a + b).hexdigest()
log("hashes done", len(res["hashes"]))
# parity rows written by the S3 job vs the arrays as written
pr = pl.read_parquet(STAGE + "parity_s3_rows.parquet")
FIELDS19 = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
            "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy"]
pk = sorted(pr["match_key"].unique().to_list())
cells = bad = 0
bad_keys = []
for mk in pk:
    m = key_m[mk]
    lo = np.searchsorted(rm_sorted, m, side="left")
    hi = np.searchsorted(rm_sorted, m, side="right")
    rr = np.sort(order[lo:hi])
    S_all = np.concatenate([np.arange(off[r], off[r + 1]) for r in rr])
    sub = pr.filter(pl.col("match_key") == mk)
    pos = np.searchsorted(S_all, sub["s"].to_numpy())
    assert (S_all[pos] == sub["s"].to_numpy()).all()
    dense = np.zeros((len(S_all), 2, 20), dtype=np.int64)
    si = sub["side_idx"].to_numpy()
    dense[pos[:, None], si[:, None], np.arange(20)[None, :]] = np.stack([sub[c].to_numpy() for c in FIELDS19 + ["equip"]], axis=1)
    got = np.zeros((len(S_all), 2, 20), dtype=np.int64)
    got[:, :, :19] = np.asarray(arr16[S_all]).astype(np.int64)
    got[:, :, 19] = np.asarray(arr_eq[S_all]).astype(np.int64)
    d = got != dense
    cells += d.shape[0] * 2
    bad += int(d.any(axis=2).sum())
    if d.any():
        bad_keys.append(mk)
res["parity_rows_vs_written"] = {"keys": len(pk), "side_steps": cells, "mismatch_side_steps": bad, "mismatch_keys": bad_keys[:10]}
log("parity vs written", res["parity_rows_vs_written"])
# ranges and row counts over the whole array
CH = 4_000_000
mx = np.full((2, 19), -10**9, dtype=np.int64)
mn = np.full((2, 19), 10**9, dtype=np.int64)
eq_mx = np.full(2, -10**9, dtype=np.int64)
rows_with_data = np.zeros(2, dtype=np.int64)
steps_with_data = 0
for i in range(0, NS, CH):
    blk = np.asarray(arr16[i:i + CH]).astype(np.int64)
    eb = np.asarray(arr_eq[i:i + CH]).astype(np.int64)
    mx = np.maximum(mx, blk.max(axis=0))
    mn = np.minimum(mn, blk.min(axis=0))
    eq_mx = np.maximum(eq_mx, eb.max(axis=0))
    nz = (blk != 0).any(axis=2) | (eb != 0)
    rows_with_data += nz.sum(axis=0)
    steps_with_data += int(nz.any(axis=1).sum())
res["ranges"] = {"min": mn.tolist(), "max": mx.tolist(), "equip_max": eq_mx.tolist()}
res["rows_with_data_CT_T"] = rows_with_data.tolist()
res["steps_with_any_data"] = steps_with_data
res["total_steps"] = NS
log("ranges and counts", json.dumps(res["ranges"]), res["rows_with_data_CT_T"], steps_with_data)
print("#RESULT " + json.dumps(res), flush=True)
