"""Paired comparison of the 13-input model with and without monotone constraints, on the same validation and test seconds.

  ...python.exe -I -X utf8 scripts\\model\\exp_mono_paired.py

Loads models/winprob_v4_basic.json (monotone) and models/winprob_v4_basic_nomono.json (no constraints), predicts every second of the
frozen validation and test matches with each model's best iteration, checks each log loss against its meta file, and reports
log loss (without constraints) minus log loss (with constraints), with 95% intervals that resample whole matches.
A negative difference means dropping the constraints is better. Writes data/combined/exp_mono_paired.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import xgboost as xgb

import train_final as T
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
MODELS = {"mono": "winprob_v4_basic", "nomono": "winprob_v4_basic_nomono"}


def predict(booster, best, X):
    return booster.predict(xgb.DMatrix(X), iteration_range=(0, best + 1))


def main():
    G, feats = T.load_basic("combined")
    sp = W.make_split("random", G, seed=0)
    va, te = sorted(sp["val"]), sorted(sp["test"])
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
    print(f"validation {len(va)} matches / {len(yva):,} seconds; test {len(te)} / {len(yte):,}", flush=True)
    out = {"inputs": len(feats), "validation_matches": len(va), "test_matches": len(te), "checks": {}, "difference": {}}
    p = {}
    for key, name in MODELS.items():
        meta = json.load(open(os.path.join(W.ROOT, "models", f"{name}.meta.json"), encoding="utf-8"))
        booster = xgb.Booster()
        booster.load_model(os.path.join(W.ROOT, "models", f"{name}.json"))
        best = int(meta["best_iteration"])
        p[key] = {"va": predict(booster, best, Xva), "te": predict(booster, best, Xte)}
        ll_te = W.evaluate(sub_te, yte, p[key]["te"])["logloss"]
        out["checks"][name] = {"best_iteration": best, "test_logloss_recomputed": ll_te, "test_logloss_meta": meta["test_logloss"]}
        print(f"{name}: best {best}, test log loss recomputed {ll_te:.6f} (meta {meta['test_logloss']:.6f})", flush=True)
    for split, sub, y in (("validation", sub_va, yva), ("test", sub_te, yte)):
        ev = W.evaluate(sub, y, p["nomono"]["va" if split == "validation" else "te"], ref=p["mono"]["va" if split == "validation" else "te"])
        out["difference"][split] = {"no_constraints_minus_constraints": ev["delta_vs_ref"], "ci95": ev["delta_vs_ref_ci95"],
                                    "matches": ev["n_matches"], "rows": ev["n_rows"]}
        print(f"{split}: no constraints minus constraints {ev['delta_vs_ref']:+.5f} {[round(x, 5) for x in ev['delta_vs_ref_ci95']]} "
              f"({ev['n_matches']} matches)", flush=True)
    W.dump_json(out, os.path.join(W.ROOT, "data", "combined", "exp_mono_paired.json"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
