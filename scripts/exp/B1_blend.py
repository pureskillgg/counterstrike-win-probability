"""B1: non-negative logit-average blend of existing per-second win-probability predictions.

Candidates: every <name>_val.npy / <name>_test.npy pair in data/viz/pt_preds whose lengths match the
validation and test label arrays. Excluded: X10_temp / X10_iso / X10_bintemp (their calibrations were fit on the
validation predictions, so their val scores are in-sample) and exact duplicates of another candidate.

Blend: p = sigmoid(sum_k w_k * logit(p_k)), w_k >= 0, sum w_k = 1, fit on VALIDATION by minimising mean logistic
loss. The objective is convex in w; solved by projected gradient with backtracking onto the simplex (numpy only).
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import sys
import glob
import json

import numpy as np

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")
import pt_eval as E

PRED = r"G:\pureskill\csds-cs2\data\viz\pt_preds"
EXCLUDE = {"X10_temp", "X10_iso", "X10_bintemp"}


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), E.EPS, 1 - E.EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def mean_logistic_loss(z, y):
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    ks = np.arange(1, len(v) + 1)
    rho = ks[u - (css - 1.0) / ks > 0][-1]
    theta = (css[rho - 1] - 1.0) / rho
    return np.maximum(v - theta, 0.0)


def fit_blend(L, y, max_iter=20000, tol=1e-13):
    K = L.shape[1]
    n = float(len(y))
    w = np.full(K, 1.0 / K)
    z = L @ w
    f = mean_logistic_loss(z, y)
    step = 1.0
    for it in range(max_iter):
        g = L.T @ (sigmoid(z) - y) / n
        while True:
            w_new = proj_simplex(w - step * g)
            d = w_new - w
            z_new = L @ w_new
            f_new = mean_logistic_loss(z_new, y)
            if f_new <= f + g @ d + (d @ d) / (2.0 * step) + 1e-18:
                break
            step *= 0.5
            if step < 1e-14:
                break
        improve = f - f_new
        if improve < 0:
            break
        w, z, f = w_new, z_new, f_new
        step = min(step * 2.0, 1e6)
        if improve < tol and it > 10:
            break
    g = L.T @ (sigmoid(z) - y) / n
    return w, f, g, it


def main():
    D = E.load()
    yv = D["val"]["y"].astype(np.float64)
    yt = D["test"]["y"].astype(np.float64)
    print(f"val rows {len(yv)}  test rows {len(yt)}", flush=True)

    ids = sorted({os.path.basename(f)[: -len("_val.npy")] for f in glob.glob(os.path.join(PRED, "*_val.npy"))})
    cands, skipped = {}, []
    for nm in ids:
        if nm in EXCLUDE:
            skipped.append((nm, "excluded: calibrated on validation"))
            continue
        if nm == "B1" or nm.startswith("B1_"):
            skipped.append((nm, "own output"))
            continue
        fv = os.path.join(PRED, nm + "_val.npy")
        ft = os.path.join(PRED, nm + "_test.npy")
        if not os.path.exists(ft):
            skipped.append((nm, "no test file"))
            continue
        pv = np.load(fv).astype(np.float64)
        pt = np.load(ft).astype(np.float64)
        if pv.shape != yv.shape or pt.shape != D["test"]["y"].shape:
            skipped.append((nm, f"length mismatch val{pv.shape} test{pt.shape}"))
            continue
        if not (np.isfinite(pv).all() and np.isfinite(pt).all()):
            skipped.append((nm, "non-finite values"))
            continue
        cands[nm] = (pv, pt)

    uniq = {}
    for nm, (pv, pt) in cands.items():
        dup = next((u for u, (uv, ut) in uniq.items() if np.array_equal(pv, uv) and np.array_equal(pt, ut)), None)
        if dup is not None:
            skipped.append((nm, f"identical to {dup}"))
        else:
            uniq[nm] = (pv, pt)
    names = list(uniq)
    print("candidates:", names, flush=True)
    for s in skipped:
        print("skipped:", s, flush=True)

    Lv = np.column_stack([logit(uniq[n][0]) for n in names])
    Lt = np.column_stack([logit(uniq[n][1]) for n in names])

    single_val = {n: float(E.ll_vec(uniq[n][0], yv).mean()) for n in names}
    single_test = {n: float(E.ll_vec(uniq[n][1], yt).mean()) for n in names}
    for n in sorted(names, key=single_val.get):
        print(f"single {n:<18} val_ll {single_val[n]:.6f}  test_ll {single_test[n]:.6f}", flush=True)

    w, f_fit, g, its = fit_blend(Lv, yv)
    print(f"fit iterations {its}  fitted val mean logistic loss {f_fit:.8f}", flush=True)
    lam = float(np.min(g[w > 1e-10])) if np.any(w > 1e-10) else float(np.min(g))
    print("KKT check (grad on active weights should equal lambda; inactive should be >= lambda):", flush=True)
    for n, wk, gk in sorted(zip(names, w, g), key=lambda t: -t[1]):
        print(f"  {n:<18} w {wk:.6f}  grad {gk:+.8f}  grad-lambda {gk - lam:+.2e}", flush=True)
    print(f"weight sum {w.sum():.12f}  min weight {w.min():.3e}", flush=True)

    pv_blend = sigmoid(Lv @ w).astype(np.float64)
    pt_blend = sigmoid(Lt @ w).astype(np.float64)
    blend_val_ll = float(E.ll_vec(pv_blend, yv).mean())
    print(f"blend val_ll (EPS-clipped, same as helper) {blend_val_ll:.6f}", flush=True)

    np.save(os.path.join(PRED, "B1_val.npy"), pv_blend)
    np.save(os.path.join(PRED, "B1_test.npy"), pt_blend)
    print("saved B1_val.npy and B1_test.npy", flush=True)

    r_blend = E.report("B1_blend", D["test"], pt_blend)

    best_val = min(names, key=single_val.get)
    best_test = min(names, key=single_test.get)
    print(f"best single by VAL: {best_val}  val_ll {single_val[best_val]:.6f}  test_ll {single_test[best_val]:.6f}", flush=True)
    print(f"best single by TEST (informational only): {best_test}  test_ll {single_test[best_test]:.6f}", flush=True)

    out = {"blend": r_blend, "weights": {n: float(x) for n, x in zip(names, w)}, "best_val": best_val,
           "best_test": best_test, "single_val": single_val, "single_test": single_test}

    for tag, nm in (("vs_best_val", best_val), ("vs_best_test", best_test)):
        Dp = dict(D["test"])
        Dp["p_base"] = uniq[nm][1]
        r = E.report(f"B1_blend_minus_{nm}", Dp, pt_blend, save=False)
        out[tag] = {"single": nm, "diff": r["vs_cpu_xgb_baseline"], "ci95": r["vs_cpu_xgb_baseline_ci95"]}
        print(f"paired diff blend - {nm} ({tag}): {r['vs_cpu_xgb_baseline']:+.6f}  ci95 {r['vs_cpu_xgb_baseline_ci95']}", flush=True)

    rb = E.report(f"B1_ref_{best_val}", D["test"], uniq[best_val][1], save=False)
    out["ref_best_val_test"] = rb
    print("SUMMARY_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
