"""Round-method comparison and hyperparameter search on the full frozen split, validation first, test second.   (XGBoost, GPU)

  ...python.exe -I -X utf8 scripts\\model\\exp_methods.py [--store combined] [--trials 16] [--stride-search 2]

Everything reads cache_snapshot_2026.parquet block by block (1,500 matches at a time), so the full table is never held in memory.
Methods (all trained on every second of the training matches unless noted; scored on every second of the validation and test matches):
  M1      one model for every round (seeds 0, 1, 2: seed spread is measured here)
  M1t     M1 plus time left in the round and on the bomb (2 extra inputs)
  M2      pistol rounds (1, 13) in their own model; every other round in a second model
  M3      pistol (1, 13), round after pistol (2, 14), and every other round, three models
  M4      economy groups: rounds split into terciles of total team equipment at round start (cut-offs from training rounds only)
  SEARCH  random hyperparameter search for M1 on every --stride-search-th second of training; the selected setting is then scored on test
Decision rule (set before the run): a grouped method (M2, M3, M4) is adopted only if its validation log loss beats M1's by more than
max(0.0002, 3 x the seed standard deviation of M1 on validation), and its test change has the same sign. Otherwise M1 stays.
Writes data/<store>/exp_methods.json after every stage.
"""
import argparse
import json
import os
import random
import sys
import time

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
ROUND_SECONDS = 115.0
BOMB_SECONDS = 41.0
BASE_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
                   tree_method="hist", seed=0)
GROUPS = {
    "all": None,
    "pistol": pl.col("round").is_in([1, 13]),
    "after_pistol": pl.col("round").is_in([2, 14]),
    "main": ~pl.col("round").is_in([1, 13, 2, 14]),
    "non_pistol": ~pl.col("round").is_in([1, 13]),
    "econ_low": pl.col("econ") == 0,
    "econ_mid": pl.col("econ") == 1,
    "econ_high": pl.col("econ") == 2,
}


def block_keys(keys, chunk=1500):
    return [keys[i:i + chunk] for i in range(0, len(keys), chunk)]


def frame(lf, ck, stride=1):
    df = lf.filter(pl.col("match_key").is_in(ck))
    if stride > 1:
        df = df.filter(pl.col("second") % stride == 0)
    return df


def to_X(df, cols, time_feats):
    X = df.select(cols).to_numpy().astype(np.float32)
    if time_feats:
        secs = df["seconds_since_round_live"].to_numpy().astype(np.float32)
        ssp = df["seconds_since_plant"].to_numpy().astype(np.float32)
        bomb = df["bomb_planted"].to_numpy()
        tl = np.maximum(0.0, ROUND_SECONDS - secs).astype(np.float32)
        bl = np.where(bomb == 1, np.maximum(0.0, BOMB_SECONDS - ssp), 0.0).astype(np.float32)
        X = np.hstack([X, np.stack([tl, bl], axis=1)])
    return X


class Blocks(xgb.DataIter):
    """Training rows of some matches, one block of matches at a time, optionally restricted to a group of rounds."""

    def __init__(self, lf, keys, cols, time_feats, group, econ_df, stride, chunk=1500):
        super().__init__()
        self.lf, self.blocks, self.cols, self.time_feats = lf, block_keys(keys, chunk), cols, time_feats
        self.group, self.econ_df, self.stride = group, econ_df, stride
        self.i = 0

    def next(self, input_data):
        while self.i < len(self.blocks):
            df = frame(self.lf, self.blocks[self.i], self.stride).collect()
            self.i += 1
            if self.econ_df is not None:
                df = df.join(self.econ_df, on=["match_key", "round"], how="left")
            if GROUPS[self.group] is not None:
                df = df.filter(GROUPS[self.group])
            if df.height == 0:
                continue
            input_data(data=to_X(df, self.cols, self.time_feats), label=df["label_ct_win"].to_numpy().astype(np.float32))
            return 1
        return 0

    def reset(self):
        self.i = 0


def full_arrays(lf, keys, cols, time_feats, econ_df):
    """Every second of some matches as arrays, with round group and economy group for routing."""
    blocks = block_keys(keys)
    n = int(sum(frame(lf, b).select(pl.len()).collect().item() for b in blocks))
    X = np.empty((n, len(cols) + (2 if time_feats else 0)), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    rnd = np.empty(n, dtype=np.int64)
    econ = np.empty(n, dtype=np.int64)
    mk = np.empty(n, dtype=object)
    w = np.empty(n, dtype=np.float64)
    i = 0
    for b in blocks:
        df = frame(lf, b).collect()
        if econ_df is not None:
            df = df.join(econ_df, on=["match_key", "round"], how="left")
        else:
            df = df.with_columns(pl.lit(-1).alias("econ"))
        m = df.height
        X[i:i + m] = to_X(df, cols, time_feats)
        y[i:i + m] = df["label_ct_win"].to_numpy()
        rnd[i:i + m] = df["round"].to_numpy()
        econ[i:i + m] = df["econ"].fill_null(-1).to_numpy()
        mk[i:i + m] = df["match_key"].to_numpy()
        w[i:i + m] = df["w_round"].to_numpy()
        i += m
        del df
    return X, y, rnd, econ, mk, w


def logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit(lf, tr, va, cols, time_feats, mono, group, econ_df, stride, params, va_arrays, device):
    """Train one model on a group of rounds. Validation for early stopping is the same group, full seconds."""
    dtr = xgb.QuantileDMatrix(Blocks(lf, tr, cols, time_feats, group, econ_df, stride), max_bin=256)
    Xva, yva, rva, eva, _, _ = va_arrays
    m = np.ones(len(yva), dtype=bool)
    if group is not None:
        expr = GROUPS[group]
        m = _mask(expr, rva, eva)
    dva = xgb.QuantileDMatrix(Xva[m], label=yva[m], ref=dtr)
    t = time.time()
    p = dict(BASE_PARAMS, **params, device=device, monotone_constraints=tuple(mono))
    bst = xgb.train(p, dtr, num_boost_round=3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    return bst, int(bst.best_iteration), round(time.time() - t), int(dtr.num_row())


def _mask(expr, rnd, econ):
    """The same group expressions, applied to arrays (round and econ columns)."""
    if expr is None:
        return np.ones(len(rnd), dtype=bool)
    df = pl.DataFrame({"round": rnd, "econ": econ})
    return df.select(expr.alias("m"))["m"].to_numpy()


def predict_routed(models, arrays, groups_used):
    """models: {group: (bst, best)}. Each row is scored by the model of its group; rows in no group get NaN."""
    X, y, rnd, econ = arrays[0], arrays[1], arrays[2], arrays[3]
    p = np.full(len(y), np.nan)
    for g in groups_used:
        bst, best = models[g]
        m = _mask(GROUPS[g], rnd, econ)
        p[m] = bst.predict(xgb.DMatrix(X[m]), iteration_range=(0, best + 1))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--stride-search", type=int, default=2)
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    out_path = os.path.join(d, "exp_methods.json")
    t0 = time.time()
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    need = list(dict.fromkeys(["match_key", "round", "second", "label_ct_win", "w_round", "seconds_since_round_live",
                               "seconds_since_plant", "bomb_planted", "st_equip_ct", "st_equip_t"] + feats))
    lf = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    mono = list(W.mono_for(feats))
    mono_t = mono + [0, 0]
    out = {"split": {"train": len(tr), "val": len(va), "test": len(te)}, "stages": {}}

    def save(stage, payload):
        out["stages"][stage] = payload
        W.dump_json(out, out_path)
        print(f"[{time.time() - t0:.0f}s] {stage} saved", flush=True)

    # economy groups: equipment at round start (second 0), cut-offs from the training rounds only
    eq = (lf.filter(pl.col("second") == 0).with_columns((pl.col("st_equip_ct") + pl.col("st_equip_t")).alias("equip0"))
          .select("match_key", "round", "equip0").collect())
    eq_tr = eq.filter(pl.col("match_key").is_in(tr))
    cuts = [float(x) for x in np.quantile(eq_tr["equip0"].to_numpy(), [1 / 3, 2 / 3])]
    econ_df = eq.with_columns(pl.when(pl.col("equip0") < cuts[0]).then(0).when(pl.col("equip0") < cuts[1]).then(1).otherwise(2)
                              .cast(pl.Int64).alias("econ")).select("match_key", "round", "econ")
    save("setup", {"econ_cutoffs_equipment_usd": cuts, "rounds_with_equipment": eq.height})

    # arrays for validation and test (full seconds); M1t uses time features as extra inputs
    va_arr = full_arrays(lf, va, feats, False, econ_df)
    te_arr = full_arrays(lf, te, feats, False, econ_df)
    print(f"val {len(va_arr[1]):,} seconds, test {len(te_arr[1]):,} seconds ({time.time() - t0:.0f}s)", flush=True)
    te_sub = pl.DataFrame({"match_key": te_arr[4].astype(str), "w_round": te_arr[5]})

    # M1: seeds 0, 1, 2
    m1 = {}
    for seed in (0, 1, 2):
        bst, best, secs, n_tr = fit(lf, tr, va, feats, False, mono, "all", None, 1, {"seed": seed}, va_arr, a.device)
        pv = bst.predict(xgb.DMatrix(va_arr[0][:, :59]), iteration_range=(0, best + 1))
        pt = bst.predict(xgb.DMatrix(te_arr[0][:, :59]), iteration_range=(0, best + 1))
        m1[f"seed{seed}"] = {"best_iteration": best, "fit_seconds": secs, "train_rows": n_tr,
                             "val_logloss": logloss(va_arr[1], pv), "test": {k: v for k, v in W.evaluate(te_sub, te_arr[1], pt).items()
                                                                              if k in ("logloss", "logloss_ci95", "logloss_round_weighted", "auc")}}
        save(f"M1_seed{seed}", m1[f"seed{seed}"])
        if seed == 0:
            p_m1_test, p_m1_val = pt, pv
    vals =[m1[f"seed{s}"]["val_logloss"] for s in (0, 1, 2)]
    seed_sd_val = float(np.std(vals, ddof=1))
    tests = [m1[f"seed{s}"]["test"]["logloss"] for s in (0, 1, 2)]
    save("M1_seed_spread", {"val_logloss_sd": seed_sd_val, "test_logloss_sd": float(np.std(tests, ddof=1)),
                            "val_mean": float(np.mean(vals)), "test_mean": float(np.mean(tests))})

    # M1t: time left in the round and on the bomb (validation and test arrays carry the same two extra inputs)
    va_arr_t = full_arrays(lf, va, feats, True, econ_df)
    te_arr_t = full_arrays(lf, te, feats, True, econ_df)
    bst, best, secs, n_tr = fit(lf, tr, va, feats, True, mono_t, "all", None, 1, {"seed": 0}, va_arr_t, a.device)
    pv = bst.predict(xgb.DMatrix(va_arr_t[0]), iteration_range=(0, best + 1))
    pt = bst.predict(xgb.DMatrix(te_arr_t[0]), iteration_range=(0, best + 1))
    save("M1t_time", {"best_iteration": best, "val_logloss": logloss(va_arr[1], pv),
                      "test": {k: v for k, v in W.evaluate(te_sub, te_arr[1], pt).items() if k in ("logloss", "logloss_ci95")}})
    del va_arr_t, te_arr_t

    def grouped(name, groups):
        models, val_p = {}, np.full(len(va_arr[1]), np.nan)
        info = {}
        for g in groups:
            bst, best, secs, n_tr = fit(lf, tr, va, feats, False, mono, g, econ_df, 1, {"seed": 0}, va_arr, a.device)
            models[g] = (bst, best)
            info[g] = {"train_rows": n_tr, "best_iteration": best, "fit_seconds": secs}
            print(f"  {name} {g}: {n_tr:,} rows, best {best} ({time.time() - t0:.0f}s)", flush=True)
        pv = predict_routed(models, va_arr, groups)
        pt = predict_routed(models, te_arr, groups)
        # rows in no group (economy groups for rounds without a freeze-end equipment row) fall back to M1 seed 0
        fb_v, fb_t = int(np.isnan(pv).sum()), int(np.isnan(pt).sum())
        pv = np.where(np.isnan(pv), p_m1_val, pv)
        pt = np.where(np.isnan(pt), p_m1_test, pt)
        rt = W.evaluate(te_sub, te_arr[1], pt)
        res = {"groups": info, "fallback_rows_val": fb_v, "fallback_rows_test": fb_t, "val_logloss": logloss(va_arr[1], pv),
               "test": {k: v for k, v in rt.items() if k in ("logloss", "logloss_ci95", "logloss_round_weighted")},
               "test_delta_vs_M1_seed0": rt["logloss"] - W.evaluate(te_sub, te_arr[1], p_m1_test)["logloss"]}
        return res, pv, pt

    res2, pv2, pt2 = grouped("M2", ["pistol", "non_pistol"])
    save("M2_pistol_split", res2)
    res3, pv3, pt3 = grouped("M3", ["pistol", "after_pistol", "main"])
    save("M3_three_split", res3)
    res4, pv4, pt4 = grouped("M4", ["econ_low", "econ_mid", "econ_high"])
    save("M4_economy_split", res4)

    # decision rule for each grouped method against M1 (seed 0 validation)
    m1_val = m1["seed0"]["val_logloss"]
    threshold = max(0.0002, 3 * seed_sd_val)
    decisions = {}
    for name, res, pv, pt in (("M2", res2, pv2, pt2), ("M3", res3, pv3, pt3), ("M4", res4, pv4, pt4)):
        gain_val = m1_val - res["val_logloss"]
        same_sign = (res["test"]["logloss"] - m1["seed0"]["test"]["logloss"]) < 0
        decisions[name] = {"validation_gain_vs_M1": gain_val, "threshold": threshold, "test_change_vs_M1": res["test"]["logloss"] - m1["seed0"]["test"]["logloss"],
                           "adopt": bool(gain_val > threshold and same_sign)}
    save("decision", decisions)

    # SEARCH: random hyperparameters for M1 on every stride-th second; selected setting scored on test
    rng = random.Random(0)
    space = {"max_depth": [3, 4, 5, 6, 8], "min_child_weight": [5, 20, 50, 100], "learning_rate": [0.03, 0.05, 0.08],
             "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.8, 1.0], "reg_lambda": [1.0, 5.0]}
    cfgs = [{"max_depth": 4, "min_child_weight": 20, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 1.0}]
    while len(cfgs) < a.trials + 1:
        c = {k: rng.choice(v) for k, v in space.items()}
        if c not in cfgs:
            cfgs.append(c)
    trials = []
    for i, c in enumerate(cfgs):
        bst, best, secs, n_tr = fit(lf, tr, va, feats, False, mono, "all", None, a.stride_search, c, va_arr, a.device)
        pv = bst.predict(xgb.DMatrix(va_arr[0][:, :59]), iteration_range=(0, best + 1))
        trials.append({"config": c, "val_logloss": logloss(va_arr[1], pv), "best_iteration": best, "fit_seconds": secs, "default": i == 0})
        save("SEARCH_trials", trials)
    best_trial = min(trials, key=lambda t: t["val_logloss"])
    default_trial = trials[0]
    pick = {}
    for name, t in (("default", default_trial), ("selected_by_validation", best_trial)):
        bst, best, secs, n_tr = fit(lf, tr, va, feats, False, mono, "all", None, 1, t["config"], va_arr, a.device)
        pt = bst.predict(xgb.DMatrix(te_arr[0][:, :59]), iteration_range=(0, best + 1))
        pick[name] = {"config": t["config"], "val_logloss": t["val_logloss"], "best_iteration": best,
                      "test": {k: v for k, v in W.evaluate(te_sub, te_arr[1], pt).items() if k in ("logloss", "logloss_ci95")}}
    save("SEARCH_selected", pick)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
