"""Paired comparison for the validation-only positions runs (pin 773d5c12).

Reads <dir>/<arm>_seed<s>.parquet for arms full_trainctr, full_oldctr, base and seeds 0 to 4, joins the arms on
(match_key, round, second or tick), checks that the labels agree row by row, and reports per-seed paired per-second log-loss
differences with the match-cluster interval from paired() in scripts/model/spatial_feature_eval.py.

    python analysis/positions_validation_paired.py --dir <positions_validation dir> --out docs/positions_validation

Bar (pinned): full_trainctr beats base if the pooled mean (mean of the five seed means) is at most -0.0005 and every seed's
upper bound is below 0; it loses if the pooled mean is at least +0.0005 and every seed's lower bound is above 0; else no claim.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "model"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from spatial_feature_eval import paired  # noqa: E402

ARMS = ["full_trainctr", "full_oldctr", "base"]
SEEDS = [0, 1, 2, 3, 4]


def load(dirpath, arm, seed):
    return pl.read_parquet(os.path.join(dirpath, f"{arm}_seed{seed}.parquet"))


def key_cols(df):
    return ["match_key", "round", "second"] if "second" in df.columns else ["match_key", "round", "tick"]


def aligned(dirpath, seed, a, b):
    da, db = load(dirpath, a, seed), load(dirpath, b, seed)
    keys = key_cols(da)
    assert keys == key_cols(db), "arms use different key columns"
    j = da.select(keys + ["label_ct_win", "p"]).rename({"p": "pa"}).join(db.select(keys + ["label_ct_win", "p"]).rename({"p": "pb"}), on=keys, how="inner")
    assert j.height == da.height == db.height, f"{a} vs {b} seed {seed}: row sets differ ({da.height}, {db.height}, joined {j.height})"
    assert (j["label_ct_win"] == j["label_ct_win_right"]).all(), "labels disagree"
    return j["match_key"].to_numpy(), j["label_ct_win"].to_numpy().astype(np.float64), j["pa"].to_numpy().astype(np.float64), j["pb"].to_numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    comparisons = [("full_trainctr", "base"), ("full_oldctr", "base"), ("full_trainctr", "full_oldctr")]
    result = {}
    for ca, cb in comparisons:
        name = f"{ca} minus {cb}"
        rows = []
        for s in SEEDS:
            match, y, pa, pb = aligned(a.dir, s, ca, cb)
            r = paired(match, y, pa, pb, seed=0)
            rows.append({"seed": s, "diff": r["diff"], "ci95": r["ci95"], "rows": r["rows"], "matches": r["matches"]})
        diffs = np.array([r["diff"] for r in rows])
        pooled = float(diffs.mean())
        result[name] = {
            "per_seed": rows,
            "pooled_mean": pooled,
            "seed_sd": float(diffs.std(ddof=1)),
            "all_upper_below_0": all(r["ci95"][1] < 0 for r in rows),
            "all_lower_above_0": all(r["ci95"][0] > 0 for r in rows),
        }
    bar = result["full_trainctr minus base"]
    if bar["pooled_mean"] <= -0.0005 and bar["all_upper_below_0"]:
        verdict = "beats base (pinned bar met)"
    elif bar["pooled_mean"] >= 0.0005 and bar["all_lower_above_0"]:
        verdict = "loses to base (pinned bar met)"
    else:
        verdict = "no claim (pinned bar not met)"
    result["verdict_full_trainctr_vs_base"] = verdict

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "positions_validation_paired.json"), "w") as f:
        json.dump(result, f, indent=2)
    lines = ["# Positions validation, paired per-second log loss (pin 773d5c12)", "",
             "Validation: 1,168 matches. Fit: 90% of the training matches; early stop: the other 10%. No test or holdout B rows used.",
             "Interval: match-cluster bootstrap, 1,000 resamples, seed 0, from paired().", ""]
    for name, r in result.items():
        if name.startswith("verdict"):
            continue
        lines += [f"## {name}", "", "| Seed | Difference | 95% interval |", "|---|---|---|"]
        for row in r["per_seed"]:
            lines.append(f"| {row['seed']} | {row['diff']:+.5f} | {row['ci95'][0]:+.5f} to {row['ci95'][1]:+.5f} |")
        lines += ["", f"Pooled mean over five seeds: {r['pooled_mean']:+.5f} (seed sd {r['seed_sd']:.5f}).", ""]
    lines += [f"Verdict, full_trainctr against base: {verdict}.", ""]
    with open(os.path.join(a.out, "positions_validation_paired.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
