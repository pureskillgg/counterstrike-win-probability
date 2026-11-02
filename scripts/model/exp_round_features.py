"""Two round-level experiments on the full frozen split, every second, the same test seconds for every model.   (XGBoost, GPU)

  ...python.exe -I -X utf8 scripts\\model\\exp_round_features.py [--store combined] [--device cuda]

base59    : the 59 snapshot inputs (reference, same settings as the final model)
time      : base59 + time left in the round (115 s live timer minus seconds since live) + time left on the bomb (41 s after the plant)
split     : round-type models, MR12 numbering. Pistol rounds 1 and 13 (first round of each half); round after pistol 2 and 14; every other round
            (3-12, 15-24 and overtime) in the main model. Each model is trained on its own rounds only, with the same 59 inputs and settings;
            each test second is scored by the model of its round. Pistol and after-pistol rounds are subtracted from the main model.
Paired differences to base59 carry a match-cluster interval; the split is also scored group by group.
Writes data/<store>/exp_round_features.json.
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

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
              tree_method="hist", seed=0)
PISTOL = (1, 13)
AFTER = (2, 14)
ROUND_SECONDS = 115.0
BOMB_SECONDS = 41.0


def fit(Xtr, ytr, Xva, yva, mono, device):
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=256)
    dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr)
    bst = xgb.train(dict(PARAMS, device=device, monotone_constraints=tuple(mono)), dtr, num_boost_round=3000, evals=[(dva, "val")],
                    early_stopping_rounds=100, verbose_eval=False)
    return bst, int(bst.best_iteration)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    t0 = time.time()
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    need = list(dict.fromkeys(["match_key", "round", "label_ct_win", "w_round", "seconds_since_round_live", "seconds_since_plant", "bomb_planted"] + feats))
    G = pl.read_parquet(os.path.join(d, "cache_snapshot_2026.parquet"), columns=need)
    G = G.with_columns([
        pl.max_horizontal(pl.lit(0.0), pl.lit(ROUND_SECONDS) - pl.col("seconds_since_round_live").cast(pl.Float64)).cast(pl.Float32).alias("time_left"),
        pl.when(pl.col("bomb_planted") == 1).then(pl.max_horizontal(pl.lit(0.0), pl.lit(BOMB_SECONDS) - pl.col("seconds_since_plant").cast(pl.Float64)))
          .otherwise(pl.lit(0.0)).cast(pl.Float32).alias("bomb_left")])
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                       pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    del kdf
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    mono = list(W.mono_for(feats))
    cols59, cols61 = feats, feats + ["time_left", "bomb_left"]
    print(f"rows loaded {G.height:,} ({time.time() - t0:.0f}s)", flush=True)

    def part(keys, cols):
        df = G.filter(pl.col("match_key").is_in(keys))
        X = df.select(cols).to_numpy().astype(np.float32)
        return df, X, df["label_ct_win"].to_numpy().astype(np.int64), df["round"].to_numpy()

    dtr, Xtr61, ytr, rtr = part(tr, cols61)
    dva, Xva61, yva, rva = part(va, cols61)
    dte, Xte61, yte, rte = part(te, cols61)
    te_sub = dte.select("match_key", "w_round")
    Xtr59, Xva59, Xte59 = Xtr61[:, :59], Xva61[:, :59], Xte61[:, :59]
    del dtr, dva
    out = {"test_matches": len(te), "test_rows": int(len(yte)), "models": {}}
    print(f"train {len(ytr):,}, val {len(yva):,}, test {len(yte):,} seconds ({time.time() - t0:.0f}s)", flush=True)

    # 1) reference: base59
    bst, best = fit(Xtr59, ytr, Xva59, yva, mono, a.device)
    p_base = bst.predict(xgb.DMatrix(Xte59), iteration_range=(0, best + 1))
    rb = W.evaluate(te_sub, yte, p_base)
    out["models"]["base59"] = {"logloss": rb["logloss"], "logloss_ci95": rb["logloss_ci95"], "logloss_round_weighted": rb["logloss_round_weighted"],
                               "best_iteration": best}
    print(f"base59: log loss {rb['logloss']:.5f} best {best} ({time.time() - t0:.0f}s)", flush=True)

    # 2) time left in the round and on the bomb
    mono61 = mono + [0, 0]
    bst, best = fit(Xtr61, ytr, Xva61, yva, mono61, a.device)
    p_time = bst.predict(xgb.DMatrix(Xte61), iteration_range=(0, best + 1))
    rt = W.evaluate(te_sub, yte, p_time, ref=p_base)
    out["models"]["time"] = {"logloss": rt["logloss"], "logloss_ci95": rt["logloss_ci95"], "logloss_round_weighted": rt["logloss_round_weighted"],
                             "best_iteration": best, "delta_vs_base59": rt["delta_vs_ref"], "delta_vs_base59_ci95": rt["delta_vs_ref_ci95"]}
    print(f"time: log loss {rt['logloss']:.5f} delta {rt['delta_vs_ref']:+.5f} {[round(x, 5) for x in rt['delta_vs_ref_ci95']]} best {best} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # 3) round-type models: pistol, after pistol, and the main model without them
    grp_tr = {"pistol": np.isin(rtr, PISTOL), "after_pistol": np.isin(rtr, AFTER)}
    grp_tr["main"] = ~(grp_tr["pistol"] | grp_tr["after_pistol"])
    grp_va = {"pistol": np.isin(rva, PISTOL), "after_pistol": np.isin(rva, AFTER)}
    grp_va["main"] = ~(grp_va["pistol"] | grp_va["after_pistol"])
    grp_te = {"pistol": np.isin(rte, PISTOL), "after_pistol": np.isin(rte, AFTER)}
    grp_te["main"] = ~(grp_te["pistol"] | grp_te["after_pistol"])
    p_split = np.full(len(yte), np.nan)
    group_models = {}
    for g in ("main", "pistol", "after_pistol"):
        t = time.time()
        bst, best = fit(Xtr59[grp_tr[g]], ytr[grp_tr[g]], Xva59[grp_va[g]], yva[grp_va[g]], mono, a.device)
        p_split[grp_te[g]] = bst.predict(xgb.DMatrix(Xte59[grp_te[g]]), iteration_range=(0, best + 1))
        group_models[g] = {"train_rows": int(grp_tr[g].sum()), "best_iteration": best, "fit_seconds": round(time.time() - t)}
        print(f"split {g}: {group_models[g]['train_rows']:,} train seconds, best {best} ({time.time() - t0:.0f}s)", flush=True)
    assert not np.isnan(p_split).any()
    rs = W.evaluate(te_sub, yte, p_split, ref=p_base)
    out["models"]["split"] = {"logloss": rs["logloss"], "logloss_ci95": rs["logloss_ci95"], "logloss_round_weighted": rs["logloss_round_weighted"],
                              "delta_vs_base59": rs["delta_vs_ref"], "delta_vs_base59_ci95": rs["delta_vs_ref_ci95"], "groups": group_models, "by_group": {}}
    te_frame = te_sub
    for g in ("pistol", "after_pistol", "main"):
        m = grp_te[g]
        rg = W.evaluate(te_frame.filter(pl.Series(m)), yte[m], p_split[m], ref=p_base[m])
        rb_g = W.evaluate(te_frame.filter(pl.Series(m)), yte[m], p_base[m])
        out["models"]["split"]["by_group"][g] = {"seconds": int(m.sum()), "logloss_split": rg["logloss"], "logloss_base59": rb_g["logloss"],
                                                  "delta_vs_base59": rg["delta_vs_ref"], "delta_vs_base59_ci95": rg["delta_vs_ref_ci95"]}
        print(f"  {g}: {int(m.sum()):,} seconds, split vs base59 {rg['delta_vs_ref']:+.5f} {[round(x, 5) for x in rg['delta_vs_ref_ci95']]}", flush=True)
    print(f"split: log loss {rs['logloss']:.5f} delta {rs['delta_vs_ref']:+.5f} {[round(x, 5) for x in rs['delta_vs_ref_ci95']]} ({time.time() - t0:.0f}s)",
          flush=True)
    W.dump_json(out, os.path.join(d, "exp_round_features.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
