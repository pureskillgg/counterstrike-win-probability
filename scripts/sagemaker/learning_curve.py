#!/usr/bin/env python3
"""Pre-registration 238a70d6: learning curve for the 10-feature event-only model on the AWS archive.

Validation only, CPU. Excludes holdout B and every match the split manifest assigns to test, before any split.
Split by match key (crc32 rules in the pin). Arms are nested prefixes of one fit order: 5,000, 10,000, and all fit matches.
Comparison: all minus each smaller arm, paired on the same validation states, match-cluster bootstrap with 1,000 resamples.
Bar: all is better if the pooled difference is at most -0.0005 and every seed's upper bound is below 0; the smaller arm is
better if the pooled difference is at least +0.0005 and every seed's lower bound is above 0; otherwise no claim.
"""
import json
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

S3 = "s3://pureskillgg-csgo-production-sagemaker-0/adam/tomes/states/event_only/"
LOCAL = Path("/mnt/sagemaker-nvme/event_only_parts")
OUT = Path("/mnt/sagemaker-nvme/learning_curve")
HB = "/home/sagemaker-user/adam_holdout_b_keys.json"
MANIFEST = "/home/sagemaker-user/csds-cs2/data/combined/split_manifest.json"
FEATS = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
         "seconds_since_plant", "round_number", "ct_score_minus_t_score", "is_pistol_round", "is_overtime"]
MONO = (1, -1, 1, 0, 0, 0, 0, 1, 0, 0)
ARMS = [("5000", 5000), ("10000", 10000), ("all", None)]
SEEDS = [0, 1, 2]


def crc_mod(s):
    return zlib.crc32(s.encode()) % 100


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", S3, str(LOCAL) + "/", "--exclude", "summary.json", "--quiet"], check=True)
    df = pl.read_parquet(str(LOCAL / "part_*.parquet"))
    print(f"rows {df.height:,}; matches {df['match_key'].n_unique():,}; loaded in {time.time() - t0:.0f}s", flush=True)

    # exclusions before any split: holdout B and the manifest's test matches
    hb = set(json.load(open(HB))["match_keys"])
    man = json.load(open(MANIFEST))
    assign = man.get("assign", man)
    test = {k for k, v in assign.items() if v == "test"}
    keys_all = set(df["match_key"].unique().to_list())
    excluded = (keys_all & hb) | (keys_all & test)
    print(f"excluded: holdout B {len(keys_all & hb)}, manifest test {len(keys_all & test)}, total {len(excluded)}",
          flush=True)
    df = df.filter(~pl.col("match_key").is_in(list(excluded)))

    # split by match key
    keys = sorted(df["match_key"].unique().to_list())
    role = {}
    for k in keys:
        if crc_mod("val:" + k) < 10:
            role[k] = "val"
        elif crc_mod("es:" + k) < 10:
            role[k] = "es"
        else:
            role[k] = "fit"
    fit_keys = sorted((k for k in keys if role[k] == "fit"), key=lambda k: zlib.crc32(("sub:" + k).encode()))
    rank = {k: i for i, k in enumerate(fit_keys)}
    n_val = sum(1 for k in keys if role[k] == "val")
    n_es = sum(1 for k in keys if role[k] == "es")
    print(f"matches after exclusion {len(keys)}: fit {len(fit_keys)}, early stop {n_es}, validation {n_val}", flush=True)

    df = df.with_columns(
        pl.col("match_key").replace_strict(role, default="none", return_dtype=pl.String).alias("role"),
        pl.col("match_key").replace_strict(rank, default=-1, return_dtype=pl.Int64).alias("rank"),
    )
    X = df.select(FEATS).to_numpy().astype(np.float32)
    y = df["label_ct_win"].to_numpy().astype(np.float32)
    role_arr = df["role"].to_numpy()
    rank_arr = df["rank"].to_numpy()
    mk = df["match_key"].to_numpy()
    va_idx = np.where(role_arr == "val")[0]
    es_idx = np.where(role_arr == "es")[0]
    fit_idx = np.where(role_arr == "fit")[0]
    print(f"rows: fit {fit_idx.size:,}, early stop {es_idx.size:,}, validation {va_idx.size:,}", flush=True)

    dva = xgb.DMatrix(X[va_idx], label=y[va_idx])
    des = xgb.DMatrix(X[es_idx], label=y[es_idx])
    yv = y[va_idx].astype(np.float64)

    results, per_row = [], {}
    for name, n in ARMS:
        sel = fit_idx if n is None else fit_idx[rank_arr[fit_idx] < n]
        dfit = xgb.DMatrix(X[sel], label=y[sel])
        n_matches = len(set(mk[sel].tolist()))
        for s in SEEDS:
            params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.03, "max_depth": 4,
                      "min_child_weight": 20, "subsample": 0.8, "tree_method": "hist", "device": "cpu",
                      "seed": s, "nthread": 16, "monotone_constraints": MONO}
            t1 = time.time()
            bst = xgb.train(params, dfit, num_boost_round=3000, evals=[(des, "es")],
                            early_stopping_rounds=100, verbose_eval=False)
            best = int(bst.best_iteration)
            p = bst.predict(dva, iteration_range=(0, best + 1)).astype(np.float64)
            ll = -(yv * np.log(np.clip(p, 1e-15, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-15, 1)))
            per_row[(name, s)] = ll
            rec = {"arm": name, "seed": s, "fit_matches": n_matches, "fit_states": int(sel.size), "best_iteration": best,
                   "validation_log_loss": float(ll.mean()), "seconds": round(time.time() - t1)}
            results.append(rec)
            print(json.dumps(rec), flush=True)

    # comparison: all minus each smaller arm, paired on the same validation states
    va_mk = mk[va_idx]
    uniq, inv = np.unique(va_mk, return_inverse=True)
    cnt = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    rng = np.random.default_rng(0)
    comparisons = {}
    for small in ["5000", "10000"]:
        seed_rows, pooled_vals = [], []
        for s in SEEDS:
            d = per_row[("all", s)] - per_row[(small, s)]  # negative means the all-data arm is better
            sums = np.bincount(inv, weights=d, minlength=uniq.size)
            boot = np.empty(1000)
            for b in range(1000):
                idx = rng.integers(0, uniq.size, uniq.size)
                boot[b] = sums[idx].sum() / cnt[idx].sum()
            mean = float(d.mean())
            ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            seed_rows.append({"seed": s, "diff_all_minus_small": mean, "ci95": ci})
            pooled_vals.append(mean)
        pooled = float(np.mean(pooled_vals))
        all_upper_below = all(r["ci95"][1] < 0 for r in seed_rows)
        all_lower_above = all(r["ci95"][0] > 0 for r in seed_rows)
        if pooled <= -0.0005 and all_upper_below:
            verdict = "all-data arm better"
        elif pooled >= 0.0005 and all_lower_above:
            verdict = f"{small}-match arm better"
        else:
            verdict = "no claim"
        comparisons[f"all_minus_{small}"] = {"pooled": pooled, "per_seed": seed_rows, "verdict": verdict}
        print(f"all minus {small}: pooled {pooled:+.6f}; {verdict}", flush=True)

    out = {"results": results, "comparisons": comparisons, "excluded": len(excluded), "fit_matches": len(fit_keys),
           "validation_matches": n_val, "seconds": round(time.time() - t0)}
    json.dump(out, open(OUT / "learning_curve.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s; wrote {OUT / 'learning_curve.json'}", flush=True)


if __name__ == "__main__":
    main()
