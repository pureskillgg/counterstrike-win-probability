"""Grid resolution for the basic model: trained on one-second rows (the shipped model) or on quarter-second rows.

Both models are scored on the SAME quarter-second validation rows (a superset of the one-second rows), and on the same quarter-second test rows.
Rule, fixed before the run: adopt the quarter-second training grid only if its validation log loss beats the one-second model by at
least 0.0005. The test set is reported, not used to choose.

  ...python.exe -I -X utf8 scripts\\model\\grid_compare.py [--device cuda]

Writes data/combined/exp_grid_step.json.
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
from exp_pistol import train
from train_final import BASIC_FEATS, MAPS

sys.stdout.reconfigure(encoding="utf-8")
RULE = 0.0005


def predict_blocks(bst, best, lf, keys, cols, chunk=300):
    """Predictions, labels, match ids and weights for some matches, one block of matches at a time. Rows are sorted by (match, tick), so
    every call returns the same rows in the same order (the anti join for defect rounds does not keep the file order)."""
    ps, ys, mks, ws = [], [], [], []
    for c0 in range(0, len(keys), chunk):
        ck = keys[c0:c0 + chunk]
        df = (lf.filter(pl.col("match_key").is_in(ck)).select(["match_key", "tick", "label_ct_win", "w_round"] + [pl.col(c).cast(pl.Float32) for c in cols])
              .collect().sort(["match_key", "tick"]))
        X = df.select(cols).to_numpy().astype(np.float32)
        ps.append(bst.predict(xgb.DMatrix(X), iteration_range=(0, best + 1)))
        ys.append(df["label_ct_win"].to_numpy().astype(np.int64))
        mks.append(df["match_key"].to_numpy())
        ws.append(df["w_round"].to_numpy())
        del df, X
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(mks), np.concatenate(ws)


def score(sub, y, p, ref=None):
    r = W.evaluate(sub, y, p, ref=ref)
    out = {"rows": int(len(y)), "matches": int(r["n_matches"]), "logloss": r["logloss"], "logloss_ci95": r["logloss_ci95"], "auc": r["auc"],
           "ece10": r["ece10"]}
    if ref is not None:
        out["delta_vs_one_second"] = r["delta_vs_ref"]
        out["delta_ci95"] = r["delta_vs_ref_ci95"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", "combined")
    t0 = time.time()
    mt = pl.read_parquet(os.path.join(d, "grid_step16_matches.parquet"))
    sp = W.make_split("random", mt, seed=0)
    ids = dict(zip(mt["match_key"].to_list(), mt["mk_id"].to_list()))
    tr = sorted(ids[k] for k in sp["train"])
    va = sorted(ids[k] for k in sp["val"])
    te = sorted(ids[k] for k in sp["test"])
    feats = BASIC_FEATS
    mono = list(W.mono_for(feats))

    lf = pl.scan_parquet(os.path.join(d, "grid_step16.parquet"))
    lf = lf.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Float32).alias(f"map_is_{m}") for m in MAPS])
    bad = pl.read_parquet(os.path.join(d, "rounds.parquet"), columns=["match_key", "round", "missing_side"]).filter(pl.col("missing_side").is_not_null())
    bad = bad.join(mt.select("match_key", "mk_id"), on="match_key", how="inner").select(pl.col("mk_id").cast(pl.Int32), pl.col("round").cast(pl.Int64))
    lf = lf.with_columns(pl.col("round").cast(pl.Int64)).join(bad.lazy(), left_on=["match_key", "round"], right_on=["mk_id", "round"], how="anti")
    print(f"matches train {len(tr)} validation {len(va)} test {len(te)} ({time.time() - t0:.0f}s)", flush=True)

    # one-second model (shipped): applied to the quarter-second rows
    meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4_basic.meta.json"), encoding="utf-8"))
    bstA = xgb.Booster()
    bstA.load_model(os.path.join(W.ROOT, "models", "winprob_v4_basic.json"))
    bestA = int(meta["best_iteration"])
    pA_va, y_va, mk_va, w_va = predict_blocks(bstA, bestA, lf, va, feats)
    pA_te, y_te, mk_te, w_te = predict_blocks(bstA, bestA, lf, te, feats)
    print(f"one-second model applied to quarter-second rows ({time.time() - t0:.0f}s)", flush=True)

    # quarter-second model
    bstB, bestB, secsB = train(lf, tr, va, feats, False, mono, None, False, {"seed": 0}, a.device)
    pB_va, yB_va, mkB_va, wB_va = predict_blocks(bstB, bestB, lf, va, feats)
    pB_te, yB_te, mkB_te, wB_te = predict_blocks(bstB, bestB, lf, te, feats)
    assert np.array_equal(y_va, yB_va) and np.array_equal(mk_va, mkB_va) and np.array_equal(w_va, wB_va), "validation rows differ between the two models"
    assert np.array_equal(y_te, yB_te) and np.array_equal(mk_te, mkB_te) and np.array_equal(w_te, wB_te), "test rows differ between the two models"
    print(f"quarter-second model: best iteration {bestB} ({secsB}s, {time.time() - t0:.0f}s)", flush=True)

    sub_va = pl.DataFrame({"match_key": mk_va, "w_round": w_va})
    sub_te = pl.DataFrame({"match_key": mk_te, "w_round": w_te})
    val = {"one_second": score(sub_va, y_va, pA_va), "quarter_second": score(sub_va, y_va, pB_va, ref=pA_va)}
    test = {"one_second": score(sub_te, y_te, pA_te), "quarter_second": score(sub_te, y_te, pB_te, ref=pA_te)}
    gain = val["quarter_second"]["delta_vs_one_second"]
    adopt = gain <= -RULE
    out = {"rule": f"adopt the quarter-second grid only if its validation log loss is at least {RULE} lower than the one-second model's",
           "validation": val, "test": test, "quarter_second_best_iteration": bestB, "adopt_quarter_second": bool(adopt),
           "decision": "quarter-second grid adopted" if adopt else "one-second grid kept"}
    W.dump_json(out, os.path.join(d, "exp_grid_step.json"))
    print(f"validation: one-second {val['one_second']['logloss']:.5f}  quarter-second {val['quarter_second']['logloss']:.5f}  "
          f"delta {gain:+.5f} {[round(x, 5) for x in val['quarter_second']['delta_ci95']]}", flush=True)
    print(f"test (reported only): one-second {test['one_second']['logloss']:.5f}  quarter-second {test['quarter_second']['logloss']:.5f}  "
          f"delta {test['quarter_second']['delta_vs_one_second']:+.5f} {[round(x, 5) for x in test['quarter_second']['delta_ci95']]}", flush=True)
    print(f"decision: {out['decision']} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
