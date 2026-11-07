import json, sys
import numpy as np
import polars as pl
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
sys.path.insert(0, S + "/scripts2")
from loadout_rule import FIELDS
E = S + "/export/"
keys = json.load(open(S + "/match_keys.json"))
off = np.load(S + "/offsets.npy").astype(np.int64)
rm = np.load(S + "/round_match.npy").astype(np.int64)
arr16 = np.load(S + "/out/loadout_step_i16.npy", mmap_mode="r")
arr_eq = np.load(S + "/out/loadout_equip_i32.npy", mmap_mode="r")
pr = pl.read_parquet(S + "/out/parity_s3_rows.parquet")
print("parity rows", pr.height, "keys", pr["match_key"].n_unique(), "sets", pr.group_by("set").agg(pl.col("match_key").n_unique()).to_dicts())
key_m = {k: i for i, k in enumerate(keys)}
res = {}
for setname in ("parity_v3", "parity_v4"):
    sub_all = pr.filter(pl.col("set") == setname)
    ok = 0
    bad = []
    cells = 0
    bad_cells = 0
    per_field = np.zeros(len(FIELDS), dtype=np.int64)
    n_keys = 0
    for mk in sorted(sub_all["match_key"].unique().to_list()):
        m = key_m[mk]
        n_keys += 1
        rr = np.flatnonzero(rm == m)
        S_all = np.concatenate([np.arange(off[r], off[r + 1]) for r in rr])
        sub = sub_all.filter(pl.col("match_key") == mk)
        pos = np.searchsorted(S_all, sub["s"].to_numpy())
        assert (S_all[pos] == sub["s"].to_numpy()).all()
        dense = np.zeros((len(S_all), 2, len(FIELDS)), dtype=np.int64)
        si = sub["side_idx"].to_numpy()
        dense[pos[:, None], si[:, None], np.arange(len(FIELDS))[None, :]] = np.stack([sub[cn].to_numpy() for cn in FIELDS], axis=1)
        loc = np.zeros((len(S_all), 2, len(FIELDS)), dtype=np.int64)
        loc[:, :, :-1] = np.asarray(arr16[S_all]).astype(np.int64)
        loc[:, :, -1] = np.asarray(arr_eq[S_all]).astype(np.int64)
        diff = loc != dense
        cells += diff.shape[0] * 2
        bad_cells += int(diff.any(axis=2).sum())
        per_field += diff.sum(axis=(0, 1))
        if not diff.any():
            ok += 1
        else:
            bad.append(mk)
    res[setname] = {"keys_compared": n_keys, "keys_all_steps_equal": ok, "side_steps_compared": cells,
                    "side_steps_mismatch": bad_cells, "mismatch_keys_first10": bad[:10],
                    "field_mismatch_counts": {f: int(c) for f, c in zip(FIELDS, per_field)}}
print(json.dumps(res, indent=1))
json.dump(res, open(S + "/out/parity_result.json", "w"), indent=1)
