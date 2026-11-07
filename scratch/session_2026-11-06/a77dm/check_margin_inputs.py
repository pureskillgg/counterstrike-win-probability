"""Descriptive check of the DM margin inputs in the sprint training matrix (fit, early-stop and validation rows).

Reads only the four columns it needs, in row chunks, from the memmap. No model is fitted and no score is computed.
Writes out/margin_check.json under the scratch folder.

    python.exe -I -X utf8 -B check_margin_inputs.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from margin_dm import margin_from_cols  # noqa: E402

MATRIX = "G:/pureskill/csds-cs2/data/sprint/matrix_trainctr"
OUT = os.path.join(HERE, "out", "margin_check.json")
CHUNK = 1_000_000


def main():
    meta = json.load(open(os.path.join(MATRIX, "meta.json")))
    feats = meta["feats_all"]
    X = np.load(os.path.join(MATRIX, "X.npy"), mmap_mode="r")
    assert X.shape == (meta["rows"], len(feats)) and X.dtype == np.float32, (X.shape, X.dtype)
    cols = {k: feats.index(k) for k in ["seconds_since_plant", "bomb_planted", "c_min_bomb_dist", "st_kit_n_ct"]}
    n_fit, n_es, n_rows = meta["n_fit"], meta["n_es"], meta["rows"]
    splits = {"fit": (0, n_fit), "early_stop": (n_fit, n_fit + n_es), "validation": (n_fit + n_es, n_rows)}
    rows_by_split = {}
    counts = {k: 0 for k in ["rows", "nan_seconds_since_plant", "nan_c_min_bomb_dist", "nan_st_kit_n_ct", "nan_margin",
                             "nan_margin_planted", "nan_margin_unplanted", "nan_sp_planted", "nan_sp_unplanted",
                             "planted_rows", "nan_margin_from_dist_or_kit_only", "nan_kit_defuse_branch"]}
    finite_chunks = []
    sp_unplanted_vals = []  # distinct non-NaN seconds_since_plant values on unplanted rows (sentinel check)
    sp_planted_min, sp_planted_max = np.inf, -np.inf
    dist_min, dist_max = np.inf, -np.inf
    kit_vals = {}
    per_split = {k: {"rows": 0, "nan_margin": 0} for k in splits}

    for name, (lo, hi) in splits.items():
        for i in range(lo, hi, CHUNK):
            j = min(i + CHUNK, hi)
            sp = np.asarray(X[i:j, cols["seconds_since_plant"]])
            bp = np.asarray(X[i:j, cols["bomb_planted"]])
            dist = np.asarray(X[i:j, cols["c_min_bomb_dist"]])
            kit = np.asarray(X[i:j, cols["st_kit_n_ct"]])
            m = margin_from_cols(sp, dist, kit)
            nan_m = np.isnan(m)
            planted = bp >= 0.5
            n = j - i
            counts["rows"] += n
            counts["nan_seconds_since_plant"] += int(np.isnan(sp).sum())
            counts["nan_c_min_bomb_dist"] += int(np.isnan(dist).sum())
            counts["nan_st_kit_n_ct"] += int(np.isnan(kit).sum())
            counts["nan_margin"] += int(nan_m.sum())
            counts["nan_margin_planted"] += int((nan_m & planted).sum())
            counts["nan_margin_unplanted"] += int((nan_m & ~planted).sum())
            counts["nan_sp_planted"] += int((np.isnan(sp) & planted).sum())
            counts["nan_sp_unplanted"] += int((np.isnan(sp) & ~planted).sum())
            counts["planted_rows"] += int(planted.sum())
            counts["nan_margin_from_dist_or_kit_only"] += int((nan_m & ~np.isnan(sp)).sum())
            counts["nan_kit_defuse_branch"] += int(np.isnan(kit).sum())
            per_split[name]["rows"] += n
            per_split[name]["nan_margin"] += int(nan_m.sum())
            unpl = sp[~planted & ~np.isnan(sp)]
            if unpl.size:
                u = np.unique(unpl)
                if len(sp_unplanted_vals) < 2000:
                    sp_unplanted_vals.extend(u[:2000].tolist())
            pl = sp[planted & ~np.isnan(sp)]
            if pl.size:
                sp_planted_min = min(sp_planted_min, float(pl.min()))
                sp_planted_max = max(sp_planted_max, float(pl.max()))
            d_ok = dist[~np.isnan(dist)]
            if d_ok.size:
                dist_min = min(dist_min, float(d_ok.min()))
                dist_max = max(dist_max, float(d_ok.max()))
            ku, kc = np.unique(kit[~np.isnan(kit)], return_counts=True)
            for k_, c_ in zip(ku.tolist(), kc.tolist()):
                kit_vals[k_] = kit_vals.get(k_, 0) + c_
            fin = m[~nan_m]
            finite_chunks.append(fin)
        print(f"{name}: {per_split[name]['rows']:,} rows, NaN margin {per_split[name]['nan_margin']:,}", flush=True)

    allfin = np.concatenate(finite_chunks).astype(np.float64)
    q = {str(p): float(np.percentile(allfin, p)) for p in [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]}
    stats = {"n_finite": int(allfin.size), "min": float(allfin.min()), "max": float(allfin.max()),
             "mean": float(allfin.mean()), "percentiles": q}
    sp_uniq = sorted(set(round(v, 6) for v in sp_unplanted_vals))
    out = {"matrix": MATRIX, "rows": meta["rows"], "splits": {k: list(v) for k, v in splits.items()},
           "counts": counts, "nan_share_margin": counts["nan_margin"] / counts["rows"],
           "nan_share_margin_by_split": {k: (v["nan_margin"] / v["rows"] if v["rows"] else None) for k, v in per_split.items()},
           "margin_finite_stats": stats,
           "seconds_since_plant_planted_range": [sp_planted_min, sp_planted_max],
           "seconds_since_plant_unplanted_distinct_non_nan_values": len(sp_uniq),
           "seconds_since_plant_unplanted_sample": sp_uniq[:20],
           "c_min_bomb_dist_range": [dist_min, dist_max],
           "st_kit_n_ct_value_counts": {str(k): v for k, v in sorted(kit_vals.items())}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
