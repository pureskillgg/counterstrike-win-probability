"""Drill into the single c_exposed exact mismatch found in bucket_c050. Read-only."""
import glob
import json

import numpy as np
import polars as pl

FIN = "/home/sagemaker-user/adam/builder/final"
V2 = FIN + "/export_clean_v2"
keys = json.load(open(V2 + "/match_keys.json"))
mid_of = {k: i for i, k in enumerate(keys)}
off = np.load(V2 + "/offsets.npy").astype(np.int64)
rm = np.load(V2 + "/round_match.npy").astype(np.int64)
rn = np.load(V2 + "/round_num.npy").astype(np.int64)
frz = np.load(V2 + "/round_freeze.npy").astype(np.int64)
pos_off = np.load(V2 + "/pos_offsets.npy").astype(np.int64)
cnt = (np.diff(off) + 1) // 2
pi8 = np.load(V2 + "/pos_int8.npy", mmap_mode="r")
ci = 8  # index of c_exposed in INT8 list
assert ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_in_contact", "t_in_contact", "c_sees_enemy",
        "t_sees_enemy", "c_exposed"][ci] == "c_exposed"

df = pl.read_parquet(f"{FIN}/pos_parity/bucket_c050.parquet")
df = df.with_columns(pl.col("match_key").replace_strict(mid_of, default=None, return_dtype=pl.Int64).alias("mid"),
                     pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
rt = pl.DataFrame({"mid": rm, "round": rn, "freeze": frz, "ridx": np.arange(len(rm), dtype=np.int64)})
j = df.join(rt, on=["mid", "round"], how="left")
a = j["c_exposed"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
ridx = j["ridx"].to_numpy().astype(np.int64)
sec = j["tick"].to_numpy().astype(np.int64) - j["freeze"].to_numpy().astype(np.int64)
jidx = sec // 128
e = pos_off[ridx] + jidx
r = np.round(a)
exp = np.clip(r, 0, 5).astype(np.int64)
got = np.asarray(pi8[e, ci]).astype(np.int64)
bad = np.where(exp != got)[0]
print("mismatch rows", len(bad))
for b in bad:
    print("parity row:", j.row(int(b), named=True))
    print("parity c_exposed float64 repr:", repr(float(a[b])), "| np.round:", float(r[b]), "| expected int8:", int(exp[b]), "| export int8:", int(got[b]))
    print("export index e:", int(e[b]), "| ridx:", int(ridx[b]), "| step j:", int(jidx[b]), "| steps in round:", int(cnt[ridx[b]]))
    lo, hi = max(0, int(e[b]) - 3), min(pi8.shape[0], int(e[b]) + 4)
    print("export c_exposed for steps e-3..e+3:", np.asarray(pi8[lo:hi, ci]).tolist())
    # other 12 int8 features at this step, for context
    print("export all int8 at e:", np.asarray(pi8[int(e[b])]).tolist())

# Check the same key in the upstream pos_parts bucket file (what the encoder read)
parts = pl.read_parquet(f"{FIN}/pos_parts/bucket_c050.parquet", columns=["match_key", "round", "tick", "c_exposed"])
mk = j[int(bad[0]), "match_key"] if len(bad) else None
rd = j[int(bad[0]), "round"] if len(bad) else None
tk = j[int(bad[0]), "tick"] if len(bad) else None
sub = parts.filter((pl.col("match_key") == mk) & (pl.col("round") == rd) & (pl.col("tick") == tk))
print("pos_parts rows for same key:", sub.height, sub.rows())
print("pos_parts row count bucket_c050:", parts.height)
