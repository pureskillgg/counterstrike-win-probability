"""verify_E1_2 (synthetic part): independent recomputation of the E1-2 claim
"CUDA and CPU predictions differ; device change alters the model".

Written by the E1-2 verifier. It does NOT import or reuse E1_bench.py / E1_parity.py code paths
beyond an identical copy of the data generator and parameters (same seed, same logit, same
order of random draws), so the numbers can be compared with logs/swarm2/E1_bench.log and
logs/swarm2/E1_parity.log.

Interpreter: the G: project env is CPU-only (USE_CUDA False). The only CUDA-enabled XGBoost on
this machine is the cs2gpu env used by E1. This script is run with that interpreter. Nothing is
installed or written into it.
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_E1_2_synth.py 100000
  ... same with 1000000

Workload (ASSUMPTION, copied from E1 scripts): 50 float32 N(0,1) features, binary:logistic target
from a nonlinear logit plus noise, seed 12345, 100k holdout rows, 200 trees, max_depth 4, eta 0.1,
hist, no subsampling, no monotone constraints. Holdout is regenerated per n_train, exactly as in E1.

Checks, all on the same data:
  A. CPU (nthread=1) on DMatrix dS, then CUDA on dS (the E1_bench order, shared DMatrix).
  B. CUDA on a fresh DMatrix dF (no earlier training on that object).
  C. CUDA repeated on a fresh DMatrix (run-to-run determinism on GPU).
  D. Cross-device prediction: CUDA model predicted on CPU, CPU model predicted on CUDA.
  E. Tree structure: split features and split thresholds per tree, first divergent tree, leaf diffs
     where the structure matches.
  F. Quantile cut values from dS (CPU) and dF (CUDA).
  G. Materiality reference: CPU model trained on a random 99% of the training rows, compared with the
     full CPU model on the same holdout (how much a tiny data change moves the model).
  H. Paired standard error of the holdout logloss gap (cuda - cpu), per-row.
"""
import json
import sys
import time

import numpy as np
import xgboost as xgb

SEED = 12345  # ASSUMPTION (same as E1)
N_FEAT = 50  # ASSUMPTION
N_HOLD = 100_000  # ASSUMPTION
N_TREES = 200  # from task
MAX_DEPTH = 4  # from task
ETA = 0.1  # ASSUMPTION


def make_data(n_train):
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


def diff_summary(a, b):
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "frac_gt_0.01": float(np.mean(d > 0.01)), "frac_gt_0.001": float(np.mean(d > 0.001)),
            "n_nonzero": int(np.count_nonzero(d)), "n": int(d.size)}


def paired_gap(p_a, p_b, y):
    """Logloss(b) - logloss(a) on the same rows, with the paired standard error."""
    la, lb = per_row_logloss(p_a, y), per_row_logloss(p_b, y)
    d = lb - la
    se = float(d.std(ddof=1) / np.sqrt(d.size))
    return {"gap_b_minus_a": float(d.mean()), "paired_se": se,
            "z": float(d.mean() / se) if se > 0 else float("nan")}


def tree_nodes(dump_json):
    trees = [json.loads(t) for t in dump_json]
    out = []
    for t in trees:
        seq = []

        def rec(node):
            if "leaf" in node:
                seq.append(("leaf", float(node["leaf"])))
            else:
                seq.append(("split", str(node["split"]), float(node["split_condition"])))
                for ch in node["children"]:
                    rec(ch)

        rec(t)
        out.append(seq)
    return out


def structure_report(b_a, b_b, label_a, label_b):
    ta = tree_nodes(b_a.get_dump(dump_format="json"))
    tb = tree_nodes(b_b.get_dump(dump_format="json"))
    n_same = 0
    first_diff = None
    leaf_max = []
    for i, (sa, sb) in enumerate(zip(ta, tb)):
        sa_s = [(x[0], x[1], x[2]) if x[0] == "split" else ("leaf",) for x in sa]
        sb_s = [(x[0], x[1], x[2]) if x[0] == "split" else ("leaf",) for x in sb]
        if sa_s == sb_s:
            n_same += 1
            la = [x[1] for x in sa if x[0] == "leaf"]
            lb = [x[1] for x in sb if x[0] == "leaf"]
            leaf_max.append(float(np.max(np.abs(np.array(la) - np.array(lb)))) if la else 0.0)
        elif first_diff is None:
            first_diff = i
    # thresholds per feature
    thr_a, thr_b = {}, {}
    for seq in ta:
        for x in seq:
            if x[0] == "split":
                thr_a.setdefault(x[1], set()).add(x[2])
    for seq in tb:
        for x in seq:
            if x[0] == "split":
                thr_b.setdefault(x[1], set()).add(x[2])
    feats = sorted(set(thr_a) | set(thr_b))
    shared = sum(len(thr_a.get(f, set()) & thr_b.get(f, set())) for f in feats)
    only_a = sum(len(thr_a.get(f, set()) - thr_b.get(f, set())) for f in feats)
    only_b = sum(len(thr_b.get(f, set()) - thr_a.get(f, set())) for f in feats)
    rep = {"n_trees": len(ta), "n_trees_same_structure": n_same,
           "first_structure_diff_tree": first_diff,
           "leaf_max_abs_diff_in_same_structure_trees_max": float(max(leaf_max)) if leaf_max else None,
           "leaf_max_abs_diff_in_same_structure_trees_median": float(np.median(leaf_max)) if leaf_max else None,
           "distinct_thresholds_shared": shared, f"distinct_thresholds_only_{label_a}": only_a,
           f"distinct_thresholds_only_{label_b}": only_b,
           "root_split_tree0": {label_a: ta[0][0][1:] if ta[0][0][0] == "split" else None,
                                label_b: tb[0][0][1:] if tb[0][0][0] == "split" else None}}
    return rep


def cut_arrays(d):
    ptr, val = d.get_quantile_cut()
    return np.asarray(ptr, dtype=np.int64), np.asarray(val, dtype=np.float64)


def cut_report(ca, cb):
    pa, va = ca
    pb, vb = cb
    na = np.diff(pa)
    nb = np.diff(pb)
    rep = {"n_features": int(len(na)), "total_cuts_cpu": int(va.size), "total_cuts_cuda": int(vb.size),
           "features_with_same_cut_count": int(np.sum(na == nb)),
           "ptr_identical": bool(np.array_equal(pa, pb)),
           "inf_cuts_cpu": int(np.sum(np.isinf(va))), "inf_cuts_cuda": int(np.sum(np.isinf(vb)))}
    if pa.shape == pb.shape and va.shape == vb.shape:
        fin = np.isfinite(va) & np.isfinite(vb)
        dif = np.abs(va[fin] - vb[fin])
        rep["finite_cuts_compared"] = int(fin.sum())
        rep["finite_cut_max_abs_diff"] = float(dif.max()) if dif.size else 0.0
        rep["finite_cut_mean_abs_diff"] = float(dif.mean()) if dif.size else 0.0
        rep["finite_cuts_identical_count"] = int(np.sum(dif == 0.0))
        rep["cut_values_identical"] = bool(np.array_equal(va, vb))
    else:
        rep["finite_cut_max_abs_diff"] = "shape mismatch"
    return rep


def run(n_train, quantized=False):
    tag = "QUANTIZED-FEATURES (round to 0.25, <256 distinct values per feature)" if quantized else "as E1 (continuous features)"
    print(f"==== n_train={n_train:,} n_hold={N_HOLD:,} trees={N_TREES} depth={MAX_DEPTH} eta={ETA} | {tag} ====", flush=True)
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)
    Xtr, ytr, Xho, yho = make_data(n_train)
    if quantized:
        # Labels stay the same (generated from the unquantized X). Only the features the model sees are rounded,
        # so every feature has few distinct values and the quantile sketch keeps all of them exactly.
        Xtr = (np.round(Xtr * 4) / 4).astype(np.float32)
        Xho = (np.round(Xho * 4) / 4).astype(np.float32)
        print("[quantized] distinct values per feature (train), min/max:",
              min(len(np.unique(Xtr[:, j])) for j in range(N_FEAT)),
              max(len(np.unique(Xtr[:, j])) for j in range(N_FEAT)), flush=True)
    print(f"[data] Xtr {Xtr.shape} {Xtr.dtype}; ytr_mean={ytr.mean():.6f}; "
          f"Xho {Xho.shape}; yho_mean={yho.mean():.6f}; Xho_sum={float(Xho.astype(np.float64).sum()):.6f}",
          flush=True)
    dho = xgb.DMatrix(Xho)

    # A. CPU then CUDA on the same DMatrix (E1_bench order)
    dS = xgb.DMatrix(Xtr, label=ytr)
    t0 = time.perf_counter()
    b_cpu = xgb.train(params("cpu", nthread=1), dS, num_boost_round=N_TREES)
    t_cpu = time.perf_counter() - t0
    p_cpu = b_cpu.predict(dho)
    cut_cpu = cut_arrays(dS)
    print(f"[A] cpu nthread=1 trained in {t_cpu:.1f}s", flush=True)

    t0 = time.perf_counter()
    b_cuda_shared = xgb.train(params("cuda"), dS, num_boost_round=N_TREES)
    t_cs = time.perf_counter() - t0
    p_cuda_shared = b_cuda_shared.predict(dho)
    print(f"[A] cuda on the CPU-used DMatrix trained in {t_cs:.1f}s", flush=True)

    # B. CUDA on a fresh DMatrix
    dF = xgb.DMatrix(Xtr, label=ytr)
    t0 = time.perf_counter()
    b_cuda = xgb.train(params("cuda"), dF, num_boost_round=N_TREES)
    t_cf = time.perf_counter() - t0
    p_cuda = b_cuda.predict(dho)
    cut_cuda = cut_arrays(dF)
    print(f"[B] cuda fresh DMatrix trained in {t_cf:.1f}s", flush=True)
    del dF

    # C. CUDA repeat on a fresh DMatrix
    dF2 = xgb.DMatrix(Xtr, label=ytr)
    b_cuda2 = xgb.train(params("cuda"), dF2, num_boost_round=N_TREES)
    p_cuda2 = b_cuda2.predict(dho)
    del dF2
    print("[C] cuda repeat done", flush=True)

    # Base score per device
    bs = {}
    for name, bst in (("cpu", b_cpu), ("cuda", b_cuda)):
        cfg = json.loads(bst.save_config())
        bs[name] = {"base_score": cfg["learner"]["learner_model_param"].get("base_score"),
                    "device": cfg["learner"]["generic_param"].get("device"),
                    "nthread": cfg["learner"]["generic_param"].get("nthread")}
    print("[meta] base_score/device per model:", json.dumps(bs), flush=True)

    # D. Cross-device prediction
    b_cuda.set_param({"device": "cpu"})
    p_cuda_on_cpu = b_cuda.predict(dho)
    b_cpu.set_param({"device": "cuda"})
    p_cpu_on_gpu = b_cpu.predict(dho)
    print("[D] cross-device prediction done", flush=True)

    res = {}
    res["logloss"] = {"cpu_nthread1": logloss(p_cpu, yho), "cuda": logloss(p_cuda, yho),
                      "cuda_shared_dmatrix": logloss(p_cuda_shared, yho), "cuda_repeat": logloss(p_cuda2, yho),
                      "cuda_model_predicted_on_cpu": logloss(p_cuda_on_cpu, yho),
                      "cpu_model_predicted_on_cuda": logloss(p_cpu_on_gpu, yho)}
    print("[logloss]", json.dumps({k: round(v, 6) for k, v in res["logloss"].items()}), flush=True)
    res["diff_cpu_vs_cuda"] = diff_summary(p_cpu, p_cuda)
    print("[diff cpu vs cuda, fresh]", json.dumps(res["diff_cpu_vs_cuda"]), flush=True)
    res["gap_cuda_minus_cpu"] = paired_gap(p_cpu, p_cuda, yho)
    print("[gap cuda-cpu, paired]", json.dumps(res["gap_cuda_minus_cpu"]), flush=True)

    res["diff_cuda_shared_vs_fresh"] = diff_summary(p_cuda_shared, p_cuda)
    print("[A vs B] cuda on shared vs fresh DMatrix:", json.dumps(res["diff_cuda_shared_vs_fresh"]), flush=True)
    res["diff_cuda_repeat_vs_fresh"] = diff_summary(p_cuda2, p_cuda)
    print("[C vs B] cuda repeat vs fresh:", json.dumps(res["diff_cuda_repeat_vs_fresh"]), flush=True)
    res["diff_cuda_model_cpu_pred_vs_native"] = diff_summary(p_cuda_on_cpu, p_cuda)
    print("[D] cuda model: CPU predictor vs native GPU predictor:",
          json.dumps(res["diff_cuda_model_cpu_pred_vs_native"]), flush=True)
    res["diff_cpu_model_gpu_pred_vs_native"] = diff_summary(p_cpu_on_gpu, p_cpu)
    print("[D] cpu model: GPU predictor vs native CPU predictor:",
          json.dumps(res["diff_cpu_model_gpu_pred_vs_native"]), flush=True)

    # E. Tree structure
    res["structure"] = structure_report(b_cpu, b_cuda, "cpu", "cuda")
    print("[E] tree structure cpu vs cuda:", json.dumps(res["structure"]), flush=True)

    # F. Cuts
    res["cuts"] = cut_report(cut_cpu, cut_cuda)
    print("[F] quantile cuts cpu vs cuda:", json.dumps(res["cuts"]), flush=True)

    # G. Materiality reference: CPU model on a random 99% of training rows
    rng = np.random.default_rng(1)
    keep = rng.random(n_train) < 0.99
    dR = xgb.DMatrix(Xtr[keep], label=ytr[keep])
    b_ref = xgb.train(params("cpu", nthread=1), dR, num_boost_round=N_TREES)
    p_ref = b_ref.predict(dho)
    del dR
    res["materiality_cpu99_vs_cpu"] = {"rows_kept": int(keep.sum()), "diff": diff_summary(p_cpu, p_ref),
                                       "gap": paired_gap(p_cpu, p_ref, yho),
                                       "logloss_cpu99": logloss(p_ref, yho)}
    print("[G] cpu on 99% rows vs cpu full:", json.dumps(res["materiality_cpu99_vs_cpu"]), flush=True)
    res["materiality_cpu99_vs_cuda"] = {"diff": diff_summary(p_ref, p_cuda),
                                        "gap": paired_gap(p_ref, p_cuda, yho)}
    print("[G] cpu on 99% rows vs cuda full:", json.dumps(res["materiality_cpu99_vs_cuda"]), flush=True)

    print("[summary-json] " + json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    quant = len(sys.argv) > 2 and sys.argv[2] == "q"
    run(n, quantized=quant)
