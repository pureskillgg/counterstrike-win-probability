"""Site-centre check, step 1: straight-line distances from site centres fitted on the TRAIN split only.

make_pos_features.site_centres() fits the two site centres per map on every match's plant positions, including the frozen test
matches. Here the same fit is repeated on train matches only (the match_keys argument), and every map's straight-line distance
features (the four columns st0..st3 used by the all-maps XGBoost arm S) are recomputed from those centres.

Parity check first: with the original centres (fitted on all matches) the distances must reproduce the saved values in
data/diagnostics/allmaps_geo/<map>.parquet. If they do not, nothing downstream is valid and the script stops.

Writes:
  data/geometry/site_centres_compare.json          original and train-only centres per map, and how far each site moved (game units)
  data/diagnostics/allmaps_geo_trainsites/<map>.parquet   match_key, round, second, st0..st3 from the train-only centres
  data/diagnostics/allmaps_geo_trainsites/parity.json     the parity check result

Run in the torch environment (ttr.build_table imports torch):
  G:/pureskill/csds-cs2/.pixi/envs/torch/python.exe -I -X utf8 -u scripts/model/sitecheck_geo.py
"""
import json
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data"))
import train_temporal_round_model as ttr  # noqa: E402
import train_temporal_allmaps as T  # noqa: E402
import make_pos_features as MPF  # noqa: E402

ROOT = ttr.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OLD = os.path.join(DIAG, "allmaps_geo")
NEW = os.path.join(DIAG, "allmaps_geo_trainsites")
GEOM = os.path.join(ROOT, "data", "geometry", "site_centres_compare.json")
N_DIST = 4


def distances(m, X, centres):
    """Same computation as train_temporal_allmaps.straight_distances, with the centres passed in.
    Columns: site A CT, site A T, site B CT, site B T; game units / 1000, capped at 10."""
    win = json.load(open(ttr.DATA / f"{m}_ext.json", encoding="utf-8"))["window"]
    lo = win["xy_min"]
    span = win["xy_max"] - win["xy_min"]
    sites = [((c[0] - lo) / span, (c[1] - lo) / span) for c in centres]
    n = X.shape[0]
    out = np.empty((n, N_DIST), dtype=np.float32)
    for c in range(0, n, 500_000):
        e = min(n, c + 500_000)
        blk = np.asarray(X[c:e])
        alive = blk[:, :, 0] > 0.5
        xy = blk[:, :, 1:3].astype(np.float64)
        for k, (sx, sy) in enumerate(sites):
            d = np.sqrt((xy[:, :, 0] - sx) ** 2 + (xy[:, :, 1] - sy) ** 2) * span
            for side, sl in enumerate([slice(0, 5), slice(5, 10)]):
                dd = np.where(alive[:, sl], d[:, sl], np.inf).min(axis=1)
                dd = np.minimum(np.where(np.isfinite(dd), dd, 10000.0), 10000.0)
                out[c:e, 2 * k + side] = dd / 1000.0
    return out


def main():
    man = json.load(open(os.path.join(ROOT, "data", "combined", "split_manifest.json"), encoding="utf-8"))["assign"]
    train_keys = [k for k, v in man.items() if v == "train"]
    print(f"train matches for the centre fit: {len(train_keys)}", flush=True)

    all_centres = MPF.site_centres("combined")
    train_centres = MPF.site_centres("combined", match_keys=train_keys)
    compare = {}
    for m in T.MAPS:
        a, t = np.asarray(all_centres[m]), np.asarray(train_centres[m])
        shift = np.linalg.norm(a - t, axis=1)
        compare[m] = {"all_matches": np.round(a, 2).tolist(), "train_only": np.round(t, 2).tolist(),
                      "site_shift_game_units": np.round(shift, 1).tolist()}
        print(f"{m}: sites moved by {np.round(shift, 1).tolist()} game units", flush=True)
    os.makedirs(os.path.dirname(GEOM), exist_ok=True)
    with open(GEOM, "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=1)

    os.makedirs(NEW, exist_ok=True)
    parity = {}
    for m in T.MAPS:
        arr, X, table, lab, st, use = ttr.build_table(m, f"{m}_ext", T.CACHE)
        keys = arr["match_keys"][arr["match_idx"]]
        st_orig = distances(m, X, all_centres[m])
        old = pl.read_parquet(os.path.join(OLD, f"{m}.parquet"))
        if old.height != st_orig.shape[0]:
            raise SystemExit(f"{m}: row count mismatch {old.height} vs {st_orig.shape[0]}")
        diff = max(float(np.max(np.abs(old[f"st{i}"].to_numpy() - st_orig[:, i]))) for i in range(N_DIST))
        parity[m] = {"rows": int(st_orig.shape[0]), "max_abs_diff_vs_saved": diff}
        print(f"{m}: parity max abs diff vs saved distances = {diff:.3g}", flush=True)
        if diff > 1e-4:
            raise SystemExit(f"{m}: parity check failed ({diff}); the original distances are not reproduced, stopping")

        st_new = distances(m, X, train_centres[m])
        df = pl.DataFrame({"match_key": keys, "round": arr["round"].astype(np.int64), "second": arr["second"].astype(np.int64)})
        df = df.with_columns([pl.Series(f"st{i}", st_new[:, i]) for i in range(N_DIST)])
        df.write_parquet(os.path.join(NEW, f"{m}.parquet"))
        print(f"{m}: {df.height:,} rows written from train-only centres", flush=True)
        del X, arr, table, st, st_orig, st_new, df, old

    with open(os.path.join(NEW, "parity.json"), "w", encoding="utf-8") as f:
        json.dump({"parity_tolerance": 1e-4, "maps": parity}, f, indent=1)
    print("done: parity passed for every map", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
