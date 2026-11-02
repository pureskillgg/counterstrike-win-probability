"""W2: walking distances with train-only site centres vs the 57-input XGBoost (all maps, three seeds, validation decision).

Pre-registration: Meridian pin f5eec3a3. Rule (standard): for EACH seed mean(W2 minus A) on validation <= -0.0005 with a 95% match-clustered
CI upper bound < 0, and the three-seed mean <= -0.0005. Test seconds are descriptive only.
Arms: A (57 frozen inputs); W2 (A plus wk0..wk3 from allmaps_walk_trainsites). Same recipe as the all-maps XGBoost (W.fit_xgb).
Writes data/diagnostics/w2_walk_trainsites.json.
Run in the cs2gpu environment (polars, xgboost, CUDA):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe -I -X utf8 -u scripts/model/w2_walk_trainsites.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp_common as W  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "w2_walk_trainsites.json")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEY = ["match_key", "round", "second"]
WK = [f"wk{i}" for i in range(4)]
SEEDS = [0, 1, 2]


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    # the walking parquet files must come from the arrays on disk now (the manifest records the source mtimes)
    walk_dir = os.path.join(DIAG, "allmaps_walk_trainsites")
    manifest = json.load(open(os.path.join(walk_dir, "manifest.json"), encoding="utf-8"))
    for m in MAPS:
        npz = os.path.join(ROOT, "data", "round_state", f"{m}_ext.npz")
        npy = os.path.join(ROOT, "data", "round_state", f"{m}_ext_navwalk_trainsites.npy")
        if manifest[m]["npz_mtime"] != os.path.getmtime(npz) or manifest[m]["npy_mtime"] != os.path.getmtime(npy):
            raise SystemExit(f"{m}: walking parquet was built from different arrays; rerun allmaps_walk_trainsites_rows.py")
    geo = pl.concat([pl.read_parquet(os.path.join(walk_dir, f"{m}.parquet")) for m in MAPS])
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"),
                        columns=sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + feats)))
    C = W.drop_roster_defect_rounds(C)
    C = C.filter(pl.col("label_ct_win").is_not_null()).with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A_ = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    print(f"rows sharing a (match_key, round, second) key: cache {int(C.select(KEY).is_duplicated().sum()):,}, "
          f"walking {int(geo.select(KEY).is_duplicated().sum()):,}; the first row of each key is kept, as in the baseline", flush=True)
    G = C.join(geo, on=KEY, how="inner").join(A_, on="match_key", how="inner")
    G = G.unique(KEY, keep="first", maintain_order=True).sort(KEY)
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter((pl.col("split") == "test") & (pl.col("map_name") == "de_mirage"))
    print(f"joined rows {G.height:,} ({time.time() - t0:.0f} s)", flush=True)
    # the decision population must be the one the pre-registered baseline (sitecheck_xgb.py) scored
    base = json.load(open(os.path.join(DIAG, "allmaps_xgb_sitecheck.json"), encoding="utf-8"))["seeds"][0]["SN_minus_A_validation"]
    n_va_matches = va["match_key"].n_unique()
    if va.height != base["rows"] or n_va_matches != base["matches"]:
        raise SystemExit(f"validation population {va.height:,} rows, {n_va_matches:,} matches; the pre-registered baseline is "
                         f"{base['rows']:,} rows and {base['matches']:,} matches")
    print(f"validation population matches the baseline: {va.height:,} rows, {n_va_matches:,} matches", flush=True)
    y_va = va["label_ct_win"].to_numpy().astype(np.int64)
    y_te = te["label_ct_win"].to_numpy().astype(np.int64)
    match_va = va["match_key"].to_numpy()
    match_te = te["match_key"].to_numpy()
    arms = {"A": feats, "W2": feats + WK}
    per = {}
    for seed in SEEDS:
        pv, pt = {}, {}
        for name, cols in arms.items():
            m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                                va.select(cols).to_numpy().astype(np.float32), y_va,
                                mono=W.mono_for(cols), seed=seed, device="cuda")
            pv[name] = W.predict(m, best, va.select(cols).to_numpy().astype(np.float32))
            pt[name] = W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))
            print(f"seed {seed} {name}: validation {float(W.ll_vec(pv[name], y_va).mean()):.6f}, "
                  f"test {float(W.ll_vec(pt[name], y_te).mean()):.6f}, best {best} ({time.time() - t0:.0f} s)", flush=True)
        per[seed] = {"W2_minus_A_validation": paired(match_va, y_va, pv["W2"], pv["A"]),
                     "W2_minus_A_test_descriptive": paired(match_te, y_te, pt["W2"], pt["A"])}
        d = per[seed]["W2_minus_A_validation"]
        print(f"seed {seed}: validation W2 minus A {d['diff']:+.6f} CI [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]", flush=True)
        W.dump_json({"seeds_done": list(per), "per_seed": per}, OUT + ".partial")  # survives a crash in a later seed
    vals = [per[s]["W2_minus_A_validation"]["diff"] for s in SEEDS]
    cis = [per[s]["W2_minus_A_validation"]["ci95"] for s in SEEDS]
    mean = float(np.mean(vals))
    per_seed_pass = [bool(v <= -0.0005 and c[1] < 0) for v, c in zip(vals, cis)]
    passes = bool(all(per_seed_pass) and mean <= -0.0005)
    res = {"per_seed": per, "per_seed_pass": per_seed_pass, "mean_over_seeds_validation": mean, "passes": passes,
           "rule": "for each seed mean <= -0.0005 with CI upper < 0, and three-seed mean <= -0.0005 (validation)",
           "seconds": round(time.time() - t0)}
    W.dump_json(res, OUT)
    print(json.dumps({"mean_over_seeds_validation": mean, "passes": passes}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
