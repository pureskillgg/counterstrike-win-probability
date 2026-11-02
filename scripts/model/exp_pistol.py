"""Can round-type handling beat one model? Per round type, on the same validation and test seconds.   (XGBoost, GPU)

  ...python.exe -I -X utf8 scripts\\model\\exp_pistol.py [--store combined] [--device cuda]

M1     one model for every round, the 59 inputs (reference)
V1     one pooled model with round-type inputs: pistol (1, 13), after pistol (2, 14), round number, overtime. Shares every row.
V5     V1 with pistol and after-pistol rounds weighted 3x in training
V2     a model for pistol rounds only, shallower and with smaller min child weight; every other round from M1
V2b    the same for rounds after pistol (2, 14)
V4     M1 with isotonic calibration fitted on the pistol validation seconds, applied to pistol test seconds
Groups: pistol (1, 13), after pistol (2, 14), rest (3-12, 15-24, overtime). Each group's log loss is reported with the paired difference to M1
on the same seconds, with a match-cluster 95% interval. Writes data/<store>/exp_pistol.json.
"""
import argparse
import json
import os
import sys
import time

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
BASE = dict(objective="binary:logistic", eval_metric="logloss", learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
            tree_method="hist", seed=0)
SMALL_GROUP = dict(max_depth=3, min_child_weight=5, learning_rate=0.03)
PISTOL = pl.col("round").is_in([1, 13])
AFTER = pl.col("round").is_in([2, 14])
REST = ~pl.col("round").is_in([1, 13, 2, 14])


def extras(df):
    r = df["round"]
    return np.stack([r.is_in([1, 13]).cast(pl.Float32).to_numpy(), r.is_in([2, 14]).cast(pl.Float32).to_numpy(),
                     r.cast(pl.Float32).to_numpy(), (r > 24).cast(pl.Float32).to_numpy()], axis=1)


class Blocks(xgb.DataIter):
    def __init__(self, lf, keys, cols, extra, group, weight_pistol, chunk=300):
        super().__init__()
        self.lf, self.cols, self.extra, self.group, self.weight_pistol = lf, cols, extra, group, weight_pistol
        self.blocks = [keys[i:i + chunk] for i in range(0, len(keys), chunk)]
        self.i = 0

    def next(self, input_data):
        while self.i < len(self.blocks):
            ck = self.blocks[self.i]
            self.i += 1
            df = self.lf.filter(pl.col("match_key").is_in(ck)).select(["round", "label_ct_win", "w_round"] + [pl.col(c).cast(pl.Float32) for c in self.cols]).collect()
            if self.group is not None:
                df = df.filter(self.group)
            if df.height == 0:
                continue
            X = df.select(self.cols).to_numpy().astype(np.float32)
            if self.extra:
                X = np.hstack([X, extras(df)])
            y = df["label_ct_win"].to_numpy().astype(np.float32)
            if self.weight_pistol:
                w = np.where(df["round"].is_in([1, 13, 2, 14]).to_numpy(), 3.0, 1.0).astype(np.float32)
                input_data(data=X, label=y, weight=w)
            else:
                input_data(data=X, label=y)
            return 1
        return 0

    def reset(self):
        self.i = 0


def arrays(lf, keys, cols, extra, chunk=1500):
    blocks = [keys[i:i + chunk] for i in range(0, len(keys), chunk)]
    frames = [lf.filter(pl.col("match_key").is_in(b)).select(["match_key", "round", "label_ct_win", "w_round"] + [pl.col(c).cast(pl.Float32) for c in cols]).collect()
              for b in blocks]
    df = pl.concat(frames)
    X = df.select(cols).to_numpy().astype(np.float32)
    if extra:
        X = np.hstack([X, extras(df)])
    return X, df["label_ct_win"].to_numpy().astype(np.int64), df["round"].to_numpy(), df["match_key"].cast(pl.Utf8), df["w_round"]


def train(lf, tr, va, cols, extra, mono, group, weight_pistol, params, device):
    dtr = xgb.QuantileDMatrix(Blocks(lf, tr, cols, extra, group, weight_pistol), max_bin=256)
    dva = xgb.QuantileDMatrix(Blocks(lf, va, cols, extra, group, False), ref=dtr)
    t = time.time()
    bst = xgb.train(dict(BASE, **params, device=device, monotone_constraints=tuple(mono)), dtr, num_boost_round=3000,
                    evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    return bst, int(bst.best_iteration), round(time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    t0 = time.time()
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    need = list(dict.fromkeys(["match_key", "round", "second", "label_ct_win", "w_round"] + feats))
    lf = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    mono = list(W.mono_for(feats))
    out = {"stages": {}}

    def save(name, payload):
        out["stages"][name] = payload
        W.dump_json(out, os.path.join(d, "exp_pistol.json"))
        print(f"[{time.time() - t0:.0f}s] {name} saved", flush=True)

    Xva, yva, rva, _, _ = arrays(lf, va, feats, False)
    Xte, yte, rte, mkte, wte = arrays(lf, te, feats, False)
    Xte_e = np.hstack([Xte, extras(pl.DataFrame({"round": rte}))])
    te_sub = pl.DataFrame({"match_key": mkte.to_list(), "w_round": wte.to_list()})
    grp = {"pistol": np.isin(rte, [1, 13]), "after": np.isin(rte, [2, 14]), "rest": ~np.isin(rte, [1, 13, 2, 14])}
    print(f"val {len(yva):,}, test {len(yte):,} seconds; test matches {len(te)} ({time.time() - t0:.0f}s)", flush=True)

    # group sizes and the no-information baseline per group (the training group mean)
    grp_tr = (lf.filter(pl.col("match_key").is_in(tr))
              .select(pl.when(pl.col("round").is_in([1, 13])).then(pl.lit("pistol")).when(pl.col("round").is_in([2, 14])).then(pl.lit("after"))
                      .otherwise(pl.lit("rest")).alias("g"), "label_ct_win")
              .group_by("g").agg(pl.len().alias("train_seconds"), pl.col("label_ct_win").mean().alias("train_CT_win_rate")).collect())
    save("groups", {r["g"]: {"train_seconds": int(r["train_seconds"]), "train_CT_win_rate": float(r["train_CT_win_rate"])} for r in grp_tr.iter_rows(named=True)})

    # M1: reference
    bst, best, s = train(lf, tr, va, feats, False, mono, None, False, {"seed": 0}, a.device)
    p_m1 = bst.predict(xgb.DMatrix(Xte), iteration_range=(0, best + 1))
    p_m1_va = bst.predict(xgb.DMatrix(Xva), iteration_range=(0, best + 1))
    save("M1", {"best_iteration": best, "fit_seconds": s})
    pred = {"M1": p_m1}

    # V1: pooled with round-type inputs
    mono_e = mono + [0, 0, 0, 0]
    bst, best, s = train(lf, tr, va, feats, True, mono_e, None, False, {"seed": 0}, a.device)
    pred["V1"] = bst.predict(xgb.DMatrix(Xte_e), iteration_range=(0, best + 1))
    save("V1_pooled_with_round_inputs", {"best_iteration": best, "fit_seconds": s})

    # V5: pooled, pistol and after-pistol rows weighted 3x
    bst, best, s = train(lf, tr, va, feats, True, mono_e, None, True, {"seed": 0}, a.device)
    pred["V5"] = bst.predict(xgb.DMatrix(Xte_e), iteration_range=(0, best + 1))
    save("V5_pooled_weighted", {"best_iteration": best, "fit_seconds": s})

    # V2: pistol-only model, tuned to small groups
    bst, best, s = train(lf, tr, va, feats, False, mono, PISTOL, False, dict(seed=0, **SMALL_GROUP), a.device)
    p = p_m1.copy()
    p[grp["pistol"]] = bst.predict(xgb.DMatrix(Xte[grp["pistol"]]), iteration_range=(0, best + 1))
    pred["V2"] = p
    save("V2_pistol_only_small", {"best_iteration": best, "fit_seconds": s, "params": SMALL_GROUP})

    # V2b: after-pistol-only model, tuned to small groups
    bst, best, s = train(lf, tr, va, feats, False, mono, AFTER, False, dict(seed=0, **SMALL_GROUP), a.device)
    p = p_m1.copy()
    p[grp["after"]] = bst.predict(xgb.DMatrix(Xte[grp["after"]]), iteration_range=(0, best + 1))
    pred["V2b"] = p
    save("V2b_after_only_small", {"best_iteration": best, "fit_seconds": s, "params": SMALL_GROUP})

    # V4: isotonic calibration of M1 on pistol validation seconds
    vmask = np.isin(rva, [1, 13])
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_m1_va[vmask], yva[vmask])
    p = p_m1.copy()
    p[grp["pistol"]] = iso.predict(p_m1[grp["pistol"]])
    pred["V4"] = p
    save("V4_calibrated_pistol", {"fitted_on_pistol_validation_seconds": int(vmask.sum())})

    # scoring: per group, paired with M1 on the same seconds
    def score(method):
        res = {}
        p = pred[method]
        for g, m in grp.items():
            rg = W.evaluate(te_sub.filter(pl.Series(m)), yte[m], p[m], ref=p_m1[m])
            res[g] = {"seconds": int(m.sum()), "logloss": rg["logloss"], "delta_vs_M1": rg["delta_vs_ref"], "delta_vs_M1_ci95": rg["delta_vs_ref_ci95"]}
        ov = W.evaluate(te_sub, yte, p, ref=p_m1)
        res["overall"] = {"logloss": ov["logloss"], "delta_vs_M1": ov["delta_vs_ref"], "delta_vs_M1_ci95": ov["delta_vs_ref_ci95"]}
        return res

    results = {m: score(m) for m in pred}
    out["results"] = results
    W.dump_json(out, os.path.join(d, "exp_pistol.json"))
    for m, res in results.items():
        line = " | ".join(f"{g} {res[g]['delta_vs_M1']:+.5f} [{res[g]['delta_vs_M1_ci95'][0]:+.5f}, {res[g]['delta_vs_M1_ci95'][1]:+.5f}]"
                          for g in ("pistol", "after", "rest"))
        print(f"{m:4s} overall {res['overall']['delta_vs_M1']:+.5f} | {line}", flush=True)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
