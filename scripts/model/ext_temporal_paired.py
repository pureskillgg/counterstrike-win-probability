"""Paired test comparison of two temporal runs on the frozen de_mirage test rows (torch env, CPU is enough).

Each arm is a prediction file in data/temporal/<pred stem>_pred.npz, written by train_temporal_round_model.py or
train_thgnn_lite.py. The test rows are rebuilt with the trainer's own build_table (the same usable mask), so both arms are
scored on the same seconds; they are joined on (match_key, round, second) and the labels must agree. Each arm's recomputed
log loss must equal its metrics file, which catches any row misalignment.

Pairing is by match (block bootstrap, 1000 resamples, seed 0), as in the other comparisons. diff = mean(A loss) - mean(B loss).

  ...torch env python.exe -I -X utf8 -u scripts\\model\\ext_temporal_paired.py --a de_mirage_ext_loadout_gru --b de_mirage_loadout_gru
Writes data/diagnostics/ext_temporal_paired_<a>_vs_<b>.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import train_temporal_round_model as ttr  # noqa: E402
import wp_common as W  # noqa: E402

ROOT = ttr.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
TEMPORAL = os.path.join(ROOT, "data", "temporal")
N_BOOT = 1000

# prediction stem -> (round-state stem, snapshot cache)
ARMS = {
    "de_mirage_loadout_gru": ("de_mirage", "cache_snapshot_2026.parquet"),
    "de_mirage_loadout_gru_ep6": ("de_mirage", "cache_snapshot_2026.parquet"),
    "de_mirage_loadout_gru_seed1": ("de_mirage", "cache_snapshot_2026.parquet"),
    "de_mirage_loadout_gru_seed2": ("de_mirage", "cache_snapshot_2026.parquet"),
    "de_mirage_loadout_gru_ms8544": ("de_mirage", "cache_snapshot_2026.parquet"),
    "de_mirage_ext_loadout_gru": ("de_mirage_ext", "cache_snapshot_2026_ext.parquet"),
    "de_mirage_ext_loadout_gru_seed1": ("de_mirage_ext", "cache_snapshot_2026_ext.parquet"),
    "de_mirage_ext_loadout_gru_seed2": ("de_mirage_ext", "cache_snapshot_2026_ext.parquet"),
    "de_mirage_ext_loadout_gru_ms8544": ("de_mirage_ext", "cache_snapshot_2026_ext.parquet"),
    "de_mirage_thgnn_lite_rerun": ("de_mirage", "cache_snapshot_2026.parquet"),
    "de_mirage_ext_thgnn_lite_rerun": ("de_mirage_ext", "cache_snapshot_2026_ext.parquet"),
    "de_mirage_thgnn_lite_ms8544": ("de_mirage", "cache_snapshot_2026.parquet"),
    "de_mirage_ext_thgnn_lite_ms8544": ("de_mirage_ext", "cache_snapshot_2026_ext.parquet"),
}


def test_rows(stem, cache, pred_stem):
    """The test rows of one arm: key, label and predicted probability, in the order of the trainer's test rows."""
    arr, X, table, label, state, usable = ttr.build_table("de_mirage", stem, cache)
    pred = np.load(os.path.join(TEMPORAL, f"{pred_stem}_pred.npz"))
    if not np.array_equal(pred["split"], arr["split"].astype(pred["split"].dtype)):
        raise SystemExit(f"{pred_stem}: split codes differ from the round-state file {stem}")
    mask = (arr["split"] == ttr.TEST) & usable
    df = pl.DataFrame({
        "match_key": table["match_key"].to_numpy()[mask],
        "round": table["round"].to_numpy()[mask],
        "second": table["second"].to_numpy()[mask],
        "label": label[mask].astype(np.int64),
        "p": pred["p"][mask].astype(np.float64),
    })
    return df


def paired(match, y, pa, pb, seed=0):
    la = W.ll_vec(pa, y)
    lb = W.ll_vec(pb, y)
    uniq, inv = np.unique(match, return_inverse=True)
    M = len(uniq)
    cnt = np.bincount(inv, minlength=M).astype(float)
    sa = np.bincount(inv, weights=la, minlength=M)
    sb = np.bincount(inv, weights=lb, minlength=M)
    rng = np.random.default_rng(seed)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, M, M), minlength=M).astype(float)
        d[b] = ((w * sa).sum() - (w * sb).sum()) / (w * cnt).sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean_a": float(la.mean()), "mean_b": float(lb.mean()), "diff": float(la.mean() - lb.mean()),
            "ci95": [float(lo), float(hi)], "rows": int(len(y)), "matches": int(M)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, choices=sorted(ARMS))
    ap.add_argument("--b", required=True, choices=sorted(ARMS))
    a = ap.parse_args()

    A = test_rows(*ARMS[a.a], a.a)
    B = test_rows(*ARMS[a.b], a.b)
    J = A.join(B, on=["match_key", "round", "second"], how="inner", suffix="_b")
    if J.height != A.height or J.height != B.height:
        raise SystemExit(f"test rows do not pair: A {A.height}, B {B.height}, joined {J.height}")
    if not (J["label"] == J["label_b"]).all():
        raise SystemExit("labels disagree on paired rows")

    def check(stem, df):
        metrics = json.load(open(os.path.join(TEMPORAL, f"{stem}_metrics.json"), encoding="utf-8"))
        ll = float(W.ll_vec(df["p"].to_numpy(), df["label"].to_numpy()).mean())
        return {"recomputed_log_loss": ll, "metrics_log_loss": metrics["test_log_loss"],
                "matches_metrics": bool(abs(ll - metrics["test_log_loss"]) < 1e-6), "rows": df.height}

    checks = {a.a: check(a.a, A), a.b: check(a.b, B)}
    y = J["label"].to_numpy()
    match = J["match_key"].to_numpy()
    res = {
        "a": a.a, "b": a.b,
        "checks": checks,
        "paired": paired(match, y, J["p"].to_numpy(), J["p_b"].to_numpy()),
        "threshold": "diff = mean(a) - mean(b); a is better when diff < 0",
    }
    os.makedirs(DIAG, exist_ok=True)
    out = os.path.join(DIAG, f"ext_temporal_paired_{a.a}_vs_{a.b}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
