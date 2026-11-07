"""Verifier child for E2-5 (CatBoost vs LightGBM CPU fit time). One fit per process.

Run with the cs2gpu_alt interpreter (catboost 1.2.10, lightgbm 4.7.0), which is the environment
that produced the E2 numbers:
  python -I -X utf8 verify_E2_5_child.py '<json spec>'
spec keys: engine ("lgb" or "cb"), protocol ("claim" or "matched"), iters (int), scratch (dir).
Prints one line starting with VERIFY_E2_5_JSON. The only disk write is a temporary CatBoost JSON
model inside the scratch directory, deleted right after it is read.

Data generator is copied from scripts\\swarm2\\E2_bench.py make_data (seed 20241, float32 features).
Timing: fit only, using time.perf_counter() around the same calls as E2_bench.py. For LightGBM the
Dataset build is timed separately (build_seconds), as in E2_bench.py. For CatBoost, fit includes
its own quantization, as in E2_bench.py. No GPU sampler thread is used here.
"""
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes

import numpy as np

SEED = 20241
N_TRAIN, N_VALID, N_FEAT = 100000, 50000, 50
THREADS = 1  # ASSUMPTION carried over from E2: 1 thread for every CPU run


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def _ft(ft):
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def sys_times():
    """(idle, busy-including-idle-kernel) in 100ns units, whole system."""
    idle, kern, user = FILETIME(), FILETIME(), FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    return _ft(idle), _ft(kern) + _ft(user)


def busy_pct(a, b):
    d_idle = b[0] - a[0]
    d_total = b[1] - a[1]
    if d_total <= 0:
        return None
    return round(100.0 * (1.0 - d_idle / d_total), 1)


def peak_rss_mb():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(pmc)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)


def make_data():
    # Copied from E2_bench.py make_data (same expression, same dtypes).
    rng = np.random.default_rng(SEED)
    n = N_TRAIN + N_VALID
    X = rng.standard_normal((n, N_FEAT), dtype=np.float32)
    f = (1.0 * X[:, 0] - 0.8 * X[:, 1] * X[:, 2] + 0.6 * np.sin(2.0 * X[:, 3])
         + 0.5 * (X[:, 4] > 0) * X[:, 5] - 0.3 * X[:, 6] ** 2 + 0.2 * X[:, 7]).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-f))
    y = (rng.random(n) < p).astype(np.int32)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def logloss(y_true, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def params_for(engine, protocol, iters):
    if engine == "lgb":
        if protocol == "claim":
            # Exactly the E2 config.
            return dict(objective="binary", learning_rate=0.1, num_leaves=63, max_bin=255, seed=0,
                        verbose=-1, device_type="cpu", num_threads=THREADS)
        # Matched attempt: depth-6 budget (<=64 leaves), no bagging, lambda_l2 and min_data_in_leaf
        # set toward CatBoost defaults. Leaf-wise growth cannot be made oblivious (symmetric).
        return dict(objective="binary", learning_rate=0.1, num_leaves=64, max_depth=6,
                    min_data_in_leaf=1, lambda_l2=3.0, max_bin=255, bagging_fraction=1.0,
                    bagging_freq=0, feature_fraction=1.0, seed=0, verbose=-1,
                    device_type="cpu", num_threads=THREADS)
    if protocol == "claim":
        # Exactly the E2 config (CatBoost defaults for boosting/bootstrap/l2 are left unset).
        return dict(iterations=iters, depth=6, learning_rate=0.1, loss_function="Logloss",
                    border_count=128, random_seed=0, verbose=0, allow_writing_files=False,
                    task_type="CPU", thread_count=THREADS)
    # Matched attempt: 255 borders, Plain boosting, no bootstrap, no random strength, L2 split score
    # (LightGBM's gain), one leaf-estimation step without backtracking (LightGBM does one Newton
    # step), and boost_from_average on (LightGBM default for binary). Defaults in the claim run were
    # MVS subsample 0.8, random_strength 1, Cosine score, leaf_estimation_iterations 10.
    return dict(iterations=iters, depth=6, learning_rate=0.1, loss_function="Logloss",
                border_count=255, random_seed=0, verbose=0, allow_writing_files=False,
                task_type="CPU", thread_count=THREADS, boosting_type="Plain",
                bootstrap_type="No", l2_leaf_reg=3.0, random_strength=0.0, rsm=1.0,
                score_function="L2", leaf_estimation_method="Newton",
                leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
                boost_from_average=True)


def run_lgb(spec, Xtr, ytr, Xva, yva, out):
    import lightgbm as lgb
    params = params_for("lgb", spec["protocol"], spec["iters"])
    out["lib_version"] = lgb.__version__
    out["params"] = params
    b0 = time.perf_counter()
    dtr = lgb.Dataset(Xtr, label=ytr, params=params, free_raw_data=False).construct()
    b1 = time.perf_counter()
    s0 = sys_times()
    t0 = time.perf_counter()
    booster = lgb.train(params, dtr, num_boost_round=spec["iters"])
    t1 = time.perf_counter()
    s1 = sys_times()
    pred = booster.predict(Xva)
    t2 = time.perf_counter()
    dump = booster.dump_model()
    leaves = [int(t["num_leaves"]) for t in dump["tree_info"]]
    out.update(build_seconds=round(b1 - b0, 4), fit_seconds=round(t1 - t0, 4),
               predict_seconds=round(t2 - t1, 4), valid_logloss=round(logloss(yva, pred), 6),
               n_trees=int(booster.num_trees()), leaves_per_tree=leaves,
               system_busy_pct_fit=busy_pct(s0, s1))


def run_cb(spec, Xtr, ytr, Xva, yva, out):
    import catboost as cb
    params = params_for("cb", spec["protocol"], spec["iters"])
    out["lib_version"] = cb.__version__
    out["params"] = params
    s0 = sys_times()
    t0 = time.perf_counter()
    model = cb.CatBoostClassifier(**params)
    model.fit(Xtr, ytr)
    t1 = time.perf_counter()
    s1 = sys_times()
    pred = model.predict_proba(Xva)[:, 1]
    t2 = time.perf_counter()
    allp = model.get_all_params()
    keys = ["boosting_type", "bootstrap_type", "border_count", "thread_count", "depth",
            "grow_policy", "l2_leaf_reg", "random_strength", "rsm", "leaf_estimation_method",
            "leaf_estimation_iterations", "score_function", "subsample", "task_type"]
    out["effective_params"] = {k: allp.get(k, "<absent>") for k in keys}
    path = os.path.join(spec["scratch"], "verify_E2_5_cb_model_%d.json" % os.getpid())
    model.save_model(path, format="json")
    with open(path, "r", encoding="utf-8") as fh:
        mdl = json.load(fh)
    os.remove(path)
    slots, nonempty = [], []
    for tr in mdl.get("oblivious_trees", []):
        lv = tr.get("leaf_values", [])
        lw = tr.get("leaf_weights")
        slots.append(len(lv))
        nonempty.append(sum(1 for w in lw if w > 0) if lw else None)
    out.update(fit_seconds=round(t1 - t0, 4), predict_seconds=round(t2 - t1, 4),
               valid_logloss=round(logloss(yva, pred), 6), n_trees=int(model.tree_count_),
               leaf_slots_per_tree=slots, nonempty_leaves_per_tree=nonempty,
               system_busy_pct_fit=busy_pct(s0, s1))


def main():
    spec = json.loads(sys.argv[1])
    out = {"engine": spec["engine"], "protocol": spec["protocol"], "iters": int(spec["iters"])}
    try:
        Xtr, ytr, Xva, yva = make_data()
        out["train_pos_rate"] = round(float(ytr.mean()), 4)
        out["valid_pos_rate"] = round(float(yva.mean()), 4)
        if spec["engine"] == "lgb":
            run_lgb(spec, Xtr, ytr, Xva, yva, out)
        else:
            run_cb(spec, Xtr, ytr, Xva, yva, out)
        out["status"] = "ok"
    except Exception as exc:  # record, do not hide
        out["status"] = "error"
        out["error_repr"] = repr(exc)[:600]
    out["peak_rss_mb"] = peak_rss_mb()
    print("VERIFY_E2_5_JSON " + json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
