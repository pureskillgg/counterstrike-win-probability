"""verify E1-2, stage B: noise-floor controls for the CPU vs CUDA holdout gap.

Question: is the CPU-vs-CUDA holdout gap larger than what harmless perturbations of the same CPU training produce?
Perturbations (ASSUMPTION: none of them is a real change of the data distribution):
  perm    : training rows shuffled (changes float summation order in the histograms; may move the sketch)
  jitter  : training features plus N(0, sd) noise, sd = 1e-4 or 1e-3 (moves the quantile cuts by about the same
            size as the observed CPU-vs-CUDA cut differences, median 1.6e-4 in the core run)
Also (100k stage): CPU thread-count invariance with nthread 1 / 4 / default on 20 trees.

Stages:
  1m   : 1M train rows, 200 trees (the claim's size). CPU orig (nthread=1), CUDA orig, 3 CPU perms (+ CUDA on same perms),
         1 CPU jitter. ~5 CPU trainings at ~55 s each.
  100k : 100k train rows, 200 trees. CPU orig, CUDA orig (should reproduce E1_parity.log), 10 CPU perms, 5 CUDA perms,
         5 CPU jitter(1e-4), 3 CPU jitter(1e-3), thread invariance on 20 trees.

Usage: <python> -I -X utf8 scripts/swarm2/verify_E1_2_controls.py <1m|100k> <scratch_dir>
"""
import faulthandler
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb

faulthandler.enable()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_E1_2_core import compare, cut_cmp, params  # noqa: E402

SEED = 12345  # ASSUMPTION (same generator as E1_bench.py / E1_parity.py)
N_HOLD = 100_000  # ASSUMPTION
N_FEAT = 50  # ASSUMPTION
N_TREES = 200  # from task


def make_data(n_train):
    """Same generator as E1_bench.make_data (n_train=1M) and E1_parity.make_data (n_train=100k)."""
    rng = np.random.default_rng(SEED)
    n = n_train + N_HOLD
    X = rng.standard_normal((n, N_FEAT), dtype=np.float32)
    logit = (
        0.8 * X[:, 0]
        - 0.6 * X[:, 1]
        + 0.5 * X[:, 2] * X[:, 3]
        + 0.4 * np.sin(2.0 * X[:, 4])
        + 0.3 * (X[:, 5] > 0)
        + 0.5 * rng.standard_normal(n).astype(np.float32)
    ).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def train(Xtr, ytr, device, n_trees, nthread=None):
    d = xgb.DMatrix(Xtr, label=ytr)
    t0 = time.perf_counter()
    b = xgb.train(params(device, nthread), d, num_boost_round=n_trees)
    dt = time.perf_counter() - t0
    cuts = None
    try:
        ptr, vals = d.get_quantile_cut()
        cuts = (np.asarray(ptr).copy(), np.asarray(vals, dtype=np.float64).copy())
    except Exception as e:  # noqa: BLE001
        print("  get_quantile_cut failed:", type(e).__name__, str(e)[:200], flush=True)
    del d
    return b, dt, cuts


ROWS = []


def record(family, name, r, cutc=None):
    row = {"family": family, "name": name, "gap": r["logloss_gap_b_minus_a"], "paired_se": r["paired_se"],
           "mean_abs": r["mean_abs"], "max_abs": r["max_abs"], "frac_gt_0.01": r["frac_abs_gt_0.01"],
           "signed_mean": r["signed_mean_b_minus_a"]}
    if cutc is not None and isinstance(cutc, dict):
        row["cut_n_differing"] = cutc.get("n_cuts_differing")
        row["cut_median_abs"] = cutc.get("median_abs_cut_diff_finite")
        row["cut_max_abs"] = cutc.get("max_abs_cut_diff_finite")
    ROWS.append(row)


def stage_1m(scratch):
    Xtr, ytr, Xho, yho = make_data(1_000_000)
    dho = xgb.DMatrix(Xho)
    pc_saved = np.load(os.path.join(scratch, "verify_E1_2_1m_cpu_pred.npy"))
    pg_saved = np.load(os.path.join(scratch, "verify_E1_2_1m_cuda_pred.npy"))

    b0, t_c, c0 = train(Xtr, ytr, "cpu", N_TREES, nthread=1)
    pc = b0.predict(dho)
    print(f"cpu orig train {t_c:.1f} s; max_abs vs stage-A saved cpu preds = "
          f"{float(np.max(np.abs(pc.astype(np.float64) - pc_saved))):.3e} (cross-process CPU determinism)", flush=True)
    bg0, t_g, cg0 = train(Xtr, ytr, "cuda", N_TREES)
    pg = bg0.predict(dho)
    print(f"cuda orig train {t_g:.1f} s; max_abs vs stage-A saved cuda preds = "
          f"{float(np.max(np.abs(pg.astype(np.float64) - pg_saved))):.3e}", flush=True)

    print("\n== REFERENCE: cpu orig vs cuda orig (1M) ==", flush=True)
    ref = compare("REF_cpu_vs_cuda", pc, pg, yho)
    record("REF", "cpu_vs_cuda", ref, cut_cmp(c0, cg0))
    print("cut cpu vs cuda:", cut_cmp(c0, cg0), flush=True)

    for k in (1, 2, 3):
        print(f"\n== CPU row permutation k={k} (1M) ==", flush=True)
        perm = np.random.default_rng(1000 + k).permutation(len(ytr))
        Xp, yp = Xtr[perm], ytr[perm]
        bp, _, cp = train(Xp, yp, "cpu", N_TREES, nthread=1)
        pp = bp.predict(dho)
        r = compare(f"noise_cpu_perm{k}_vs_cpu_orig", pc, pp, yho)
        cc = cut_cmp(c0, cp)
        print("cut cpu_orig vs cpu_perm:", cc, flush=True)
        record("cpu_perm_1m", f"perm{k}", r, cc)
        compare(f"cross_cpu_perm{k}_vs_cuda_orig", pp, pg, yho)

        bgp, _, cgp = train(Xp, yp, "cuda", N_TREES)
        pgp = bgp.predict(dho)
        rg = compare(f"noise_cuda_perm{k}_vs_cuda_orig", pg, pgp, yho)
        record("cuda_perm_1m", f"perm{k}", rg, cut_cmp(cg0, cgp))
        print("cut cuda_orig vs cuda_perm:", cut_cmp(cg0, cgp), flush=True)
        del Xp, yp

    print("\n== CPU feature jitter sd=1e-4 (1M) ==", flush=True)
    rj = np.random.default_rng(777)
    Xj = Xtr + np.float32(1e-4) * rj.standard_normal(Xtr.shape, dtype=np.float32)
    bj, _, cj = train(Xj, ytr, "cpu", N_TREES, nthread=1)
    pj = bj.predict(dho)
    r = compare("noise_cpu_jitter1e-4_vs_cpu_orig", pc, pj, yho)
    cc = cut_cmp(c0, cj)
    print("cut cpu_orig vs cpu_jitter:", cc, flush=True)
    record("cpu_jitter_1m", "sd1e-4", r, cc)
    compare("cross_cpu_jitter_vs_cuda_orig", pj, pg, yho)

    print("\nROWS_JSON_BEGIN", flush=True)
    print(json.dumps(ROWS, indent=1, default=str), flush=True)
    print("ROWS_JSON_END", flush=True)


def summarize(rows, fam):
    sel = [r for r in rows if r["family"] == fam]
    if not sel:
        return None
    gaps = np.array([r["gap"] for r in sel])
    ma = np.array([r["mean_abs"] for r in sel])
    return {"family": fam, "n": len(sel), "gap_mean": float(gaps.mean()),
            "gap_sd": float(gaps.std(ddof=1)) if len(sel) > 1 else None,
            "gap_min": float(gaps.min()), "gap_max": float(gaps.max()),
            "mean_abs_min": float(ma.min()), "mean_abs_max": float(ma.max())}


def stage_100k(scratch):
    Xtr, ytr, Xho, yho = make_data(100_000)
    dho = xgb.DMatrix(Xho)
    b0, t_c, c0 = train(Xtr, ytr, "cpu", N_TREES, nthread=1)
    pc = b0.predict(dho)
    bg0, t_g, cg0 = train(Xtr, ytr, "cuda", N_TREES)
    pg = bg0.predict(dho)
    print(f"cpu train {t_c:.1f} s, cuda train {t_g:.1f} s", flush=True)
    print("\n== REFERENCE 100k: cpu vs cuda (E1_parity.log: logloss 0.595975 / 0.596233, gap +0.000258, "
          "mean_abs 0.018632, frac>0.01 0.6375) ==", flush=True)
    ref = compare("REF100k_cpu_vs_cuda", pc, pg, yho)
    record("REF", "cpu_vs_cuda_100k", ref, cut_cmp(c0, cg0))
    print("cut cpu vs cuda 100k:", cut_cmp(c0, cg0), flush=True)

    for k in range(1, 11):
        perm = np.random.default_rng(2000 + k).permutation(len(ytr))
        bp, _, cp = train(Xtr[perm], ytr[perm], "cpu", N_TREES, nthread=1)
        pp = bp.predict(dho)
        r = compare(f"noise_cpu_perm{k}_vs_cpu_orig", pc, pp, yho)
        record("cpu_perm_100k", f"perm{k}", r, cut_cmp(c0, cp))

    for k in range(1, 6):
        perm = np.random.default_rng(2000 + k).permutation(len(ytr))
        bgp, _, cgp = train(Xtr[perm], ytr[perm], "cuda", N_TREES)
        pgp = bgp.predict(dho)
        r = compare(f"noise_cuda_perm{k}_vs_cuda_orig", pg, pgp, yho)
        record("cuda_perm_100k", f"perm{k}", r, cut_cmp(cg0, cgp))

    for sd in (1e-4, 1e-3):
        for k in range(1, 4 if sd == 1e-3 else 6):
            rj = np.random.default_rng(3000 + k)
            Xj = Xtr + np.float32(sd) * rj.standard_normal(Xtr.shape, dtype=np.float32)
            bj, _, cj = train(Xj, ytr, "cpu", N_TREES, nthread=1)
            pj = bj.predict(dho)
            r = compare(f"noise_cpu_jitter{sd:g}_k{k}_vs_cpu_orig", pc, pj, yho)
            record(f"cpu_jitter_100k_sd{sd:g}", f"k{k}", r, cut_cmp(c0, cj))
            del Xj

    print("\n== thread invariance (100k, 20 trees): exact prediction equality ==", flush=True)
    th = {}
    for name, nth in (("nthread1", 1), ("nthread4", 4), ("default", None)):
        b, dt, _ = train(Xtr, ytr, "cpu", 20, nthread=nth)
        th[name] = (b.predict(dho), dt)
        print(f"  {name}: {dt:.2f} s", flush=True)
    base = th["nthread1"][0].astype(np.float64)
    for name in ("nthread4", "default"):
        d = np.abs(th[name][0].astype(np.float64) - base)
        print(f"  {name} vs nthread1: max_abs={float(d.max()):.3e} exact_equal={bool(np.array_equal(th[name][0], th['nthread1'][0]))}",
              flush=True)

    print("\n== SUMMARY 100k families (gap = logloss(b) - logloss(a) against cpu_orig unless noted) ==", flush=True)
    for fam in ("REF", "cpu_perm_100k", "cuda_perm_100k", "cpu_jitter_100k_sd0.0001", "cpu_jitter_100k_sd0.001"):
        print(json.dumps(summarize(ROWS, fam), default=str), flush=True)
    print("ROWS_JSON_BEGIN", flush=True)
    print(json.dumps(ROWS, indent=1, default=str), flush=True)
    print("ROWS_JSON_END", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    scratch = sys.argv[2]
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), "stage", stage, flush=True)
    if stage == "1m":
        stage_1m(scratch)
    elif stage == "100k":
        stage_100k(scratch)
    else:
        raise SystemExit(f"unknown stage {stage}")
