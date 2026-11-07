"""Check the patched freeze-tick loadout in export_clean_v4 directly (read-only on v2 and v4)."""
import json
import numpy as np
import polars as pl

V2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"

names = json.load(open(V2 + "loadout_names.json"))
ia_ct = names.index("CT_alive")
ia_t = names.index("T_alive")
keys = json.load(open(V2 + "match_keys.json"))
kid = {k: i for i, k in enumerate(keys)}
rm = np.load(V2 + "round_match.npy").astype(np.int64)
rn = np.load(V2 + "round_num.npy").astype(np.int64)
L2 = np.load(V2 + "loadout_round.npy", mmap_mode="r")
L4 = np.load(V4 + "loadout_round.npy", mmap_mode="r")
print("shapes", L2.shape, L4.shape, "names", len(names))

ov = pl.read_csv(FX + "override_rounds.csv", infer_schema_length=0).with_columns(pl.col("round").cast(pl.Int64))
fixed = ov.filter(pl.col("status") == "fixed")

ok_ct = ok_t = 0
bad = []
fixed_idx = []
for row in fixed.iter_rows(named=True):
    mk = row["match_key"]
    r = int(row["round"])
    idx = np.where((rm == kid[mk.rsplit("/", 1)[0].rsplit("/", 0)[0]] if False else rm == -1))[0]  # placeholder
    fixed_idx.append((mk, r))

# map (match_key, round) -> row index using match_keys order: keys holds the S3 folder dir + /csds
def key_for(mk):
    return mk


lookup = {}
for i in range(len(rm)):
    lookup[(keys[int(rm[i])], int(rn[i]))] = i

changed = 0
for mk, r in fixed_idx:
    i = lookup.get((mk, r))
    if i is None:
        bad.append((mk, r, "no row"))
        continue
    ct_alive = float(L4[i, ia_ct])
    t_alive = float(L4[i, ia_t])
    if ct_alive == 5:
        ok_ct += 1
    else:
        bad.append((mk, r, "CT", ct_alive))
    if t_alive == 5:
        ok_t += 1
    else:
        bad.append((mk, r, "T", t_alive))

# every row that differs from v2 must be one of the fixed rounds
diff_rows = np.where(np.any(np.asarray(L2) != np.asarray(L4), axis=1))[0]
fixed_rows = set(lookup[(mk, r)] for mk, r in fixed_idx if (mk, r) in lookup)
extra = [int(i) for i in diff_rows if int(i) not in fixed_rows]
print("fixed rounds", len(fixed_idx), "mapped", len(fixed_rows))
print("CT alive==5:", ok_ct, " T alive==5:", ok_t)
print("rows differing from v2:", len(diff_rows), " of which outside fixed rounds:", len(extra))
print("bad:", bad[:5], "n_bad", len(bad))
