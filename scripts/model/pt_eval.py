"""Shared evaluation for per-second win-probability experiments. numpy only (works in any env).

    import sys; sys.path.insert(0, r"G:\\pureskill\\csds-cs2\\scripts\\model")
    import pt_eval as E
    D = E.load()                               # D["train"], D["val"], D["test"] -> dicts of arrays
    res = E.report("my_model", D["test"], p_test)   # prints and appends to data/viz/pt_results.jsonl
Intervals are match-cluster bootstrap (1000 resamples, seed 0). Paired differences are against the CPU XGBoost
grid baseline stored in test.npz as p_base (log loss 0.4916).
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import json
from pathlib import Path

import numpy as np

CACHE = Path(r"G:\pureskill\csds-cs2\data\viz\pt_cache")
RESULTS = Path(r"G:\pureskill\csds-cs2\data\viz\pt_results.jsonl")
EPS = 1e-15
N_BOOT = 1000


def load():
    return {k: dict(np.load(CACHE / f"{k}.npz")) for k in ("train", "val", "test")}


def ll_vec(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(y, p):
    y = np.asarray(y).astype(bool)
    order = np.argsort(p, kind="mergesort")
    ps = np.asarray(p)[order]
    _, start, cnt = np.unique(ps, return_index=True, return_counts=True)
    avg = start + (cnt + 1) / 2.0
    ranks = np.empty(len(ps))
    ranks[order] = np.repeat(avg, cnt)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece10(y, p):
    b = np.minimum((np.asarray(p) * 10).astype(int), 9)
    e = 0.0
    for i in range(10):
        m = b == i
        if m.any():
            e += m.mean() * abs(float(np.mean(y[m])) - float(np.mean(np.asarray(p)[m])))
    return float(e)


def _ratio_boot(num, den, mid, idx):
    M = idx.shape[1]
    sn = np.bincount(mid, weights=num, minlength=M)
    sd = np.bincount(mid, weights=den, minlength=M)
    return sn[idx].sum(axis=1) / np.maximum(sd[idx].sum(axis=1), 1e-12), float(sn.sum() / sd.sum())


def _pct(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def report(name, test, p, extra=None, save=True):
    y = test["y"].astype(np.int64)
    mid = test["mid"].astype(np.int64)
    wr = test["wr"].astype(np.float64)
    M = int(mid.max()) + 1
    idx = np.random.default_rng(0).integers(0, M, size=(N_BOOT, M))
    one = np.ones(len(y))
    ll = ll_vec(p, y)
    rs, pt = _ratio_boot(ll, one, mid, idx)
    rr, ptr = _ratio_boot(ll * wr, wr, mid, idx)
    d = ll - ll_vec(test["p_base"], y)
    ds, dpt = _ratio_boot(d, one, mid, idx)
    out = {"name": name, "n": int(len(y)), "logloss": pt, "logloss_ci95": _pct(rs), "logloss_round_weighted": ptr,
           "brier": float(np.mean((np.asarray(p) - y) ** 2)), "auc": auc(y, p), "ece10": ece10(y, p),
           "vs_cpu_xgb_baseline": dpt, "vs_cpu_xgb_baseline_ci95": _pct(ds)}
    if extra:
        out.update(extra)
    print(json.dumps(out), flush=True)
    if save:
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
    return out
