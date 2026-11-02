"""Close-out checks for the methodology: three numbers the draft left as holes.

  1. Like-for-like: the 13-input model (winprob_v4_basic) on the 2026 test matches, the same matches the 59-input model is scored on
     (its 3,059), against that model's 0.45641 on those matches.
  2. Tick cap: rows for rounds with no bomb planted at 115 to 116 seconds of live play. The round clock is 1.92 minutes (115.2 s);
     the grid cap is 116 s. Counts the rows in that band, their outcome and their predictions, and the test score with and without them.
  3. Reconciliation: the matches in the split manifest, the 2026 snapshot cache and the grid (2026 era), and the ones that differ.

  ...python.exe -I -X utf8 scripts\\model\\exp_closeout_checks.py
Writes data/combined/closeout_checks.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import train_final as T
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    d = os.path.join(W.ROOT, "data", "combined")
    out = {}
    assign = json.load(open(os.path.join(d, "split_manifest.json"), encoding="utf-8"))["assign"]
    booster = xgb.Booster()
    booster.load_model(os.path.join(W.ROOT, "models", "winprob_v4_basic.json"))
    best = int(json.load(open(os.path.join(W.ROOT, "models", "winprob_v4_basic.meta.json"), encoding="utf-8"))["best_iteration"])
    meta59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))

    def predict(X):
        return booster.predict(xgb.DMatrix(X), iteration_range=(0, best + 1))

    G, feats = T.load_basic("combined")  # 13 inputs, every era, roster-defect rounds dropped (as in the served model's test)
    print(f"grid rows after the roster drop: {G.height:,}", flush=True)

    # 1. like-for-like on the 2026 test matches
    cache_keys = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select("match_key").unique().collect()["match_key"].to_list()
    test26 = sorted(k for k in cache_keys if assign.get(k) == "test")
    sub, X, y = W.arrays(G, test26, feats, stride=1)
    ev = W.evaluate(sub, y, predict(X))
    out["like_for_like"] = {
        "test_matches_2026_in_cache": len(test26), "matches_scored_13_inputs": ev["n_matches"], "rows_13_inputs": ev["n_rows"],
        "logloss_13_inputs": ev["logloss"], "logloss_13_inputs_ci95": ev["logloss_ci95"],
        "logloss_59_inputs_same_matches_reported": meta59["test_logloss"], "difference_13_minus_59": ev["logloss"] - meta59["test_logloss"],
        "note": "the 59-input figure is on its own rows for these matches; no interval on the difference (not paired)"}
    print(f"like-for-like: 13 inputs {ev['logloss']:.5f} on {ev['n_matches']} matches vs 59 inputs {meta59['test_logloss']:.5f}", flush=True)

    # 2. the 115-to-116-second band for rounds with no bomb planted
    te_grid = sorted(W.make_split("random", G, seed=0)["test"])
    sub_t, Xt, yt = W.arrays(G, te_grid, feats, stride=1)
    pt = predict(Xt)
    s = sub_t["seconds_since_round_live"].to_numpy()
    unpl = sub_t["bomb_planted"].to_numpy() == 0
    band = unpl & (s >= 115) & (s < 116)
    beyond = unpl & (s >= 116)
    cap = {"test_rows": int(len(yt)), "unplanted_rows_115_to_116": int(band.sum()), "unplanted_rows_at_or_beyond_116": int(beyond.sum()),
           "max_seconds_unplanted": float(s[unpl].max()) if unpl.any() else None}
    if band.any():
        cap["matches_with_band_rows"] = int(sub_t.filter(pl.Series(band))["match_key"].n_unique())
        cap["ct_win_rate_in_band"] = float(yt[band].mean())
        cap["mean_prediction_in_band"] = float(pt[band].mean())
        cap["band_logloss"] = W.evaluate(sub_t.filter(pl.Series(band)), yt[band], pt[band])["logloss"]
        cap["logloss_all_test_rows"] = W.evaluate(sub_t, yt, pt)["logloss"]
        cap["logloss_without_band_rows"] = W.evaluate(sub_t.filter(pl.Series(~band)), yt[~band], pt[~band])["logloss"]
    out["tick_cap"] = cap
    print(f"cap band: {cap['unplanted_rows_115_to_116']:,} unplanted rows at 115 to 116 s; beyond 116 s: {cap['unplanted_rows_at_or_beyond_116']:,}", flush=True)

    # 3. reconciliation of the match sets
    raw = (pl.scan_parquet(os.path.join(d, "grid.parquet")).select(["match_key", "era", "round"]).group_by(["match_key", "era"])
           .agg(pl.len().alias("rows"), pl.col("round").n_unique().alias("rounds")).collect())
    grid_all = set(raw["match_key"].to_list())
    grid26 = raw.filter(pl.col("era") == "2026")
    grid26_keys = set(grid26["match_key"].to_list())
    cache_set = set(cache_keys)
    manifest = set(assign)
    detail = {r["match_key"]: r for r in raw.iter_rows(named=True)}

    def describe(keys):
        return [{"match_key": k, "split": assign.get(k), "grid_era": (detail[k]["era"] if k in detail else None),
                 "grid_rows": (detail[k]["rows"] if k in detail else 0), "grid_rounds": (detail[k]["rounds"] if k in detail else 0)}
                for k in sorted(keys)]

    grid26_not_cache = grid26_keys - cache_set
    cache_not_grid26 = cache_set - grid26_keys
    manifest_not_grid = manifest - grid_all
    out["reconcile"] = {
        "manifest_matches": len(manifest), "cache_2026_matches": len(cache_set), "cache_in_manifest": len(cache_set & manifest),
        "grid_matches_all_eras": len(grid_all), "grid_2026_matches": len(grid26_keys), "grid_2026_and_cache": len(grid26_keys & cache_set),
        "grid_2026_not_in_cache": describe(grid26_not_cache), "cache_not_in_grid_2026": describe(cache_not_grid26),
        "manifest_not_in_grid_any_era": describe(manifest_not_grid),
    }
    print(f"manifest {len(manifest):,}; cache {len(cache_set):,}; grid 2026 {len(grid26_keys):,}; grid 2026 and cache {len(grid26_keys & cache_set):,}; "
          f"grid 2026 not in cache {len(grid26_not_cache)}; cache not in grid 2026 {len(cache_not_grid26)}; "
          f"manifest not in grid {len(manifest_not_grid)}", flush=True)
    W.dump_json(out, os.path.join(d, "closeout_checks.json"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
