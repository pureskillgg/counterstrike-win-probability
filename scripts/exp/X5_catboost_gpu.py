"""X5: CatBoost (GPU, fallback CPU) win-probability model. Depth 6 vs 8 picked on validation.

Run with the CatBoost env (numpy only, no pandas/sklearn):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu_alt-16300978804608365045/envs/default/python.exe -I -X utf8 this_file
"""
import os
import sys
import time

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")

import numpy as np  # noqa: E402
import pt_eval as E  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

PRED_DIR = r"G:\pureskill\csds-cs2\data\viz\pt_preds"
ID = "X5"
MONO = [1, -1, 1, 0, 0, 0, 0, 1, 0, 0]
DEPTHS = [6, 8]
OOM_RETRIES = 3
OOM_WAIT_S = 30

notes = []
T0 = time.time()

D = E.load()
Xtr, ytr = D["train"]["X"], D["train"]["y"]
Xva, yva = D["val"]["X"], D["val"]["y"]
Xte = D["test"]["X"]
print(f"train {Xtr.shape} val {Xva.shape} test {Xte.shape}", flush=True)


def is_oom(exc):
    s = str(exc).lower()
    return ("out of memory" in s) or ("cuda" in s and "memory" in s)


def fit_once(depth, device, mono):
    params = dict(
        loss_function="Logloss",
        depth=depth,
        learning_rate=0.08,
        iterations=3000,
        early_stopping_rounds=100,
        random_seed=0,
        allow_writing_files=False,
        verbose=500,
    )
    if device == "GPU":
        params.update(task_type="GPU", devices="0", gpu_ram_part=0.2)
    else:
        params.update(thread_count=4)
    if mono:
        params["monotone_constraints"] = MONO
    m = CatBoostClassifier(**params)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    return m


def fit_with_retry(depth, device, mono):
    """Retry CUDA OOM up to OOM_RETRIES times, 30 s apart. Other errors propagate."""
    for attempt in range(OOM_RETRIES + 1):
        try:
            return fit_once(depth, device, mono)
        except Exception as exc:  # noqa: BLE001
            if device == "GPU" and is_oom(exc) and attempt < OOM_RETRIES:
                msg = f"depth {depth} GPU OOM on attempt {attempt + 1}, waiting {OOM_WAIT_S}s: {str(exc)[:200]}"
                print(msg, flush=True)
                notes.append(msg)
                time.sleep(OOM_WAIT_S)
                continue
            raise
    raise RuntimeError("unreachable")


def fit_depth(depth):
    """Try GPU with monotone constraints; then GPU without; then CPU with monotone constraints."""
    plan = [("GPU", True), ("GPU", False), ("CPU", True)]
    for device, mono in plan:
        t = time.time()
        try:
            m = fit_with_retry(depth, device, mono)
            pv = m.predict_proba(Xva)[:, 1].astype(np.float64)
            vll = float(E.ll_vec(pv, yva).mean())
            msg = (f"depth {depth} OK device={device} monotone={mono} best_iter={m.get_best_iteration()} "
                   f"val_logloss_rowmean={vll:.6f} fit_s={time.time() - t:.1f}")
            print(msg, flush=True)
            notes.append(msg)
            return m, device, mono, pv, vll
        except Exception as exc:  # noqa: BLE001
            msg = f"depth {depth} FAILED device={device} monotone={mono}: {type(exc).__name__}: {str(exc)[:300]}"
            print(msg, flush=True)
            notes.append(msg)
    raise RuntimeError(f"all device plans failed for depth {depth}")


results = {}
for depth in DEPTHS:
    results[depth] = fit_depth(depth)

best_depth = min(DEPTHS, key=lambda d: results[d][4])
m, dev, mono, p_val, vll = results[best_depth]
p_test = m.predict_proba(Xte)[:, 1].astype(np.float64)
p_val = np.asarray(p_val, dtype=np.float64)

os.makedirs(PRED_DIR, exist_ok=True)
np.save(os.path.join(PRED_DIR, f"{ID}_val.npy"), p_val)
np.save(os.path.join(PRED_DIR, f"{ID}_test.npy"), p_test)
print(f"saved {ID}_val.npy {p_val.shape} and {ID}_test.npy {p_test.shape}", flush=True)

extra = {
    "depth": int(best_depth),
    "device": dev,
    "monotone_constraints": bool(mono),
    "best_iteration": int(m.get_best_iteration()),
    "val_logloss_rowmean": float(vll),
    "val_logloss_by_depth": {str(d): float(results[d][4]) for d in DEPTHS},
    "gpu_ram_part": 0.2,
    "fit_total_s": round(time.time() - T0, 1),
}
res = E.report(ID, D["test"], p_test, extra=extra)
print("NOTES:", " | ".join(notes), flush=True)
print(f"TOTAL_S {time.time() - T0:.1f}", flush=True)
