"""verify E1-2, stage A: recompute the 1M-row CPU vs CUDA holdout numbers from scratch.

Same synthetic generator, seed, and XGBoost parameters as scripts/swarm2/E1_bench.py (all ASSUMPTION, copied from it).
Each booster is trained on a fresh DMatrix. Alternative-explanation checks in this stage:
  - CUDA run-to-run determinism (train CUDA twice, compare predictions exactly)
  - predictor-only effect (CPU-trained model predicted by CUDA predictor, and the reverse)
  - base_score and signed mean difference (systematic shift?)
  - first differing tree and root split (where do the two models diverge?)
  - quantile cut comparison (the claim's suspected cause)
  - paired standard error of the holdout logloss gap

Run with the C: cs2gpu interpreter (CUDA build). The G: env is CPU-only (see verify_E1_2.log).
Usage: <python> -I -X utf8 scripts/swarm2/verify_E1_2_core.py <scratch_dir>
"""
import faulthandler
import json
import os
import re
import sys
import time

import numpy as np
import xgboost as xgb

faulthandler.enable()

SEED = 12345  # ASSUMPTION (same as E1_bench.py)
N_TRAIN = 1_000_000  # ASSUMPTION (same as E1_bench.py)
N_HOLD = 100_000  # ASSUMPTION
N_FEAT = 50  # ASSUMPTION
N_TREES = 200  # from task
MAX_DEPTH = 4  # from task
ETA = 0.1  # ASSUMPTION


def make_data():
    rng = np.random.default_rng(SEED)
    n = N_TRAIN + N_HOLD
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
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def params(device, nthread=None):
    p = {"objective": "binary:logistic", "tree_method": "hist", "device": device,
         "max_depth": MAX_DEPTH, "eta": ETA, "seed": 0}
    if nthread is not None:
        p["nthread"] = nthread
    return p


def per_row_logloss(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    yy = np.asarray(y, dtype=np.float64)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def logloss(p, y):
    return float(per_row_logloss(p, y).mean())


def logit_of(p):
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return np.log(q / (1 - q))


def train(Xtr, ytr, device, nthread=None):
    d = xgb.DMatrix(Xtr, label=ytr)
    t0 = time.perf_counter()
    b = xgb.train(params(device, nthread), d, num_boost_round=N_TREES)
    dt = time.perf_counter() - t0
    cuts = None
    try:
        ptr, vals = d.get_quantile_cut()
        cuts = (np.asarray(ptr).copy(), np.asarray(vals, dtype=np.float64).copy())
    except Exception as e:  # noqa: BLE001
        print("  get_quantile_cut failed:", type(e).__name__, str(e)[:200], flush=True)
    del d
    return b, dt, cuts


def compare(name, pa, pb, yho):
    a = np.asarray(pa, dtype=np.float64)
    b = np.asarray(pb, dtype=np.float64)
    d = b - a
    ad = np.abs(d)
    gap_rows = per_row_logloss(b, yho) - per_row_logloss(a, yho)
    se = float(gap_rows.std(ddof=1) / np.sqrt(len(gap_rows)))
    gap = float(gap_rows.mean())
    ldiff = np.abs(logit_of(b) - logit_of(a))
    # elementwise Pearson r (np.corrcoef -> BLAS crashed this process with 0xc06d007f on this host)
    ca_, cb_ = a - a.mean(), b - b.mean()
    corr = float((ca_ * cb_).sum() / np.sqrt((ca_ * ca_).sum() * (cb_ * cb_).sum()))
    res = {
        "logloss_a": logloss(a, yho), "logloss_b": logloss(b, yho),
        "logloss_gap_b_minus_a": gap, "paired_se": se, "t_stat": gap / se if se > 0 else float("nan"),
        "max_abs": float(ad.max()), "mean_abs": float(ad.mean()), "frac_abs_gt_0.01": float(np.mean(ad > 0.01)),
        "signed_mean_b_minus_a": float(d.mean()), "logit_max_abs": float(ldiff.max()),
        "logit_mean_abs": float(ldiff.mean()), "pearson_r": corr, "n": int(len(a)),
    }
    print(f"[{name}] logloss_a={res['logloss_a']:.6f} logloss_b={res['logloss_b']:.6f} "
          f"gap(b-a)={gap:+.6f} paired_se={se:.6f} t={res['t_stat']:+.2f}", flush=True)
    print(f"[{name}] max_abs={res['max_abs']:.6f} mean_abs={res['mean_abs']:.6f} "
          f"frac>0.01={res['frac_abs_gt_0.01']:.4f} signed_mean={res['signed_mean_b_minus_a']:+.6f} "
          f"logit_max={res['logit_max_abs']:.4f} logit_mean={res['logit_mean_abs']:.4f} r={corr:.6f}",
          flush=True)
    return res


def cut_cmp(ca, cb):
    """Compare quantile cut arrays. Each feature's cut list ends with a non-finite sentinel (inf), so
    non-finite entries are counted separately and excluded from the magnitude stats."""
    if ca is None or cb is None:
        return "cut unavailable"
    pa, va = ca
    pb, vb = cb
    if not np.array_equal(pa, pb):
        return f"indptr differ: lens {len(pa)} vs {len(pb)}"
    fin = np.isfinite(va) & np.isfinite(vb)
    same = (va == vb) | (np.isnan(va) & np.isnan(vb))
    d = np.abs(va[fin] - vb[fin])
    return {"n_cuts": int(len(va)), "n_nonfinite_a": int((~np.isfinite(va)).sum()),
            "n_nonfinite_b": int((~np.isfinite(vb)).sum()), "n_finite_pairs": int(fin.sum()),
            "n_cuts_differing": int((~same).sum()),
            "max_abs_cut_diff_finite": float(d.max()) if d.size else None,
            "median_abs_cut_diff_finite": float(np.median(d)) if d.size else None}


def tree_cmp(ta, tb):
    n_same = sum(1 for x, y in zip(ta, tb) if x == y)
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    out = {"n_trees": len(ta), "identical_trees_text": n_same, "first_differing_tree": first}
    if first is not None:
        la, lb = ta[first].splitlines(), tb[first].splitlines()
        diffs = [(x, y) for x, y in zip(la, lb) if x != y][:3]
        out["first_diff_lines_example"] = diffs
    # split feature per node (depth-first order), ignoring thresholds
    fa = [re.findall(r"\[f(\d+)<", t) for t in ta]
    fb = [re.findall(r"\[f(\d+)<", t) for t in tb]
    out["trees_same_feature_sequence"] = int(sum(1 for x, y in zip(fa, fb) if x == y))
    tot = sum(len(x) for x in fa)
    same_nodes = sum(sum(1 for u, v in zip(x, y) if u == v) for x, y in zip(fa, fb))
    out["split_nodes_total"] = int(tot)
    out["split_nodes_same_feature_same_position"] = int(same_nodes)
    return out


def base_score(b):
    cfg = json.loads(b.save_config())
    return cfg["learner"]["learner_model_param"].get("base_score")


if __name__ == "__main__":
    scratch = sys.argv[1]
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)
    print("python", sys.executable, flush=True)
    t0 = time.perf_counter()
    Xtr, ytr, Xho, yho = make_data()
    print(f"data generated in {time.perf_counter() - t0:.2f} s; Xtr {Xtr.shape} {Xtr.dtype} "
          f"{Xtr.nbytes / 2**20:.1f} MiB; Xho {Xho.shape}", flush=True)
    dho = xgb.DMatrix(Xho)

    bc, tc, cc = train(Xtr, ytr, "cpu", nthread=1)
    print(f"cpu nthread=1 train {tc:.1f} s", flush=True)
    pc = bc.predict(dho)
    bg, tg, cg = train(Xtr, ytr, "cuda")
    print(f"cuda train {tg:.1f} s", flush=True)
    pg = bg.predict(dho)
    bg2, tg2, cg2 = train(Xtr, ytr, "cuda")
    print(f"cuda run 2 train {tg2:.1f} s", flush=True)
    pg2 = bg2.predict(dho)
    os.makedirs(scratch, exist_ok=True)
    np.save(os.path.join(scratch, "verify_E1_2_1m_cpu_pred.npy"), pc)
    np.save(os.path.join(scratch, "verify_E1_2_1m_cuda_pred.npy"), pg)
    np.save(os.path.join(scratch, "verify_E1_2_1m_yho.npy"), yho)
    print("predictions saved to scratch", flush=True)

    print("\n== claim numbers (bench reported cpu 0.590320, cuda 0.595471; max 0.2006283, mean 0.0191946) ==", flush=True)
    r_main = compare("cpu1thread_vs_cuda", pc, pg, yho)

    print("\n== CUDA determinism ==", flush=True)
    det = {"max_abs_run1_vs_run2": float(np.max(np.abs(pg.astype(np.float64) - pg2.astype(np.float64)))),
           "exact_equal": bool(np.array_equal(pg, pg2))}
    print("cuda run1 vs run2:", det, flush=True)

    print("\n== predictor-only effect (same model, other predictor) ==", flush=True)
    bcx = bc.copy()
    bcx.set_param({"device": "cuda"})
    pc_on_gpu = bcx.predict(dho)
    bgx = bg.copy()
    bgx.set_param({"device": "cpu"})
    pg_on_cpu = bgx.predict(dho)
    pred_only = {"cpu_model_gpu_predictor_vs_cpu_predictor": compare("cpu_model:cpu_pred_vs_gpu_pred", pc, pc_on_gpu, yho),
                 "cuda_model_cpu_predictor_vs_gpu_pred": compare("cuda_model:gpu_pred_vs_cpu_pred", pg, pg_on_cpu, yho)}

    print("\n== base_score and tree structure ==", flush=True)
    print("base_score cpu:", base_score(bc), " cuda:", base_score(bg), flush=True)
    ta, tb = bc.get_dump(dump_format="text", with_stats=False), bg.get_dump(dump_format="text", with_stats=False)
    tcmp = tree_cmp(ta, tb)
    print("tree_cmp:", json.dumps(tcmp, indent=1, default=str), flush=True)
    print("root line cpu:", ta[0].splitlines()[0], flush=True)
    print("root line cuda:", tb[0].splitlines()[0], flush=True)

    print("\n== quantile cuts (claim's suspected cause) ==", flush=True)
    print("cut cpu vs cuda:", cut_cmp(cc, cg), flush=True)
    print("cut cuda run1 vs run2:", cut_cmp(cg, cg2), flush=True)

    summary = {"main": r_main, "determinism": det, "predictor_only": pred_only,
               "base_score": {"cpu": base_score(bc), "cuda": base_score(bg)}, "tree_cmp": tcmp,
               "cut_cpu_vs_cuda": cut_cmp(cc, cg), "cut_cuda_run1_vs_run2": cut_cmp(cg, cg2),
               "train_s": {"cpu1": tc, "cuda1": tg, "cuda2": tg2}}
    print("\nSUMMARY_JSON_BEGIN", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("SUMMARY_JSON_END", flush=True)
