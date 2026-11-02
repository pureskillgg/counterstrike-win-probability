"""Merge parts written by exp_full_comparison.py --out into data/combined/exp_full_comparison.json.

  ...python.exe -I -X utf8 scripts\\model\\merge_comparison.py exp_full_comparison_part_a.json exp_full_comparison_part_b.json

Every part must have been run on the same split (same match and row counts) and must carry the same XGBoost baseline
(its test log loss is compared). Model rows are copied as they are; the rankings are recomputed over all of them the same way
exp_full_comparison.py does it. Parts are not deleted.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
SHARED = ["inputs", "stride", "train_matches", "val_matches", "test_matches", "train_rows", "val_rows", "test_rows"]


def main():
    d = os.path.join(W.ROOT, "data", "combined")
    parts = [json.load(open(os.path.join(d, name), encoding="utf-8")) for name in sys.argv[1:]]
    if len(parts) < 2:
        sys.exit("give at least two part files")
    base = parts[0]
    for p in parts[1:]:
        for k in SHARED:
            if p[k] != base[k]:
                sys.exit(f"parts disagree on {k}: {base[k]} vs {p[k]}")
    xgb_tests = {round(p["models"]["xgb"]["logloss"], 9) for p in parts if "xgb" in p["models"]}
    if len(xgb_tests) != 1:
        sys.exit(f"parts carry different XGBoost baselines: {xgb_tests}")
    models = {}
    for p in parts:
        for k, v in p["models"].items():
            if k in models:
                # the XGBoost baseline is in every part (each part's deltas are measured against it); keep one copy
                if k == "xgb" and abs(models[k]["logloss"] - v["logloss"]) < 1e-9:
                    continue
                sys.exit(f"model {k} appears in two parts with different results")
            models[k] = v
    ll = sorted(models, key=lambda k: models[k]["logloss"])
    rw = sorted(models, key=lambda k: models[k]["logloss_round_weighted"])
    for i, k in enumerate(ll):
        models[k]["rank_per_second"] = i + 1
        models[k]["rank_round_weighted"] = rw.index(k) + 1
    out = {k: base[k] for k in SHARED}
    out["models"] = models
    out["ranking_per_second"] = ll
    out["ranking_round_weighted"] = rw
    out["merged_from"] = [os.path.basename(n) for n in sys.argv[1:]]
    W.dump_json(out, os.path.join(d, "exp_full_comparison.json"))
    print("per second:", ll)
    print("round-weighted:", rw)


if __name__ == "__main__":
    main()
