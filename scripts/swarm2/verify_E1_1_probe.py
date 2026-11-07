"""verify_E1_1 probe: independent check of "CUDA xgboost works in the C: env".

Uses its own synthetic data (seed 2026, 20 features), not the E1 benchmark data.
All numbers are printed; nothing is written to disk.

Modes (first argv, default "full"):
  full         build flags + shared library path; CUDA train on 1M x 20, 50 trees with
               nvidia-smi sampling; then a child process with CUDA_VISIBLE_DEVICES=-1 as a
               no-fallback control (a silent CPU fallback would still succeed there).
  tiny         build flags; tiny cpu and cuda train attempts (control for the G: CPU env).
  child_nogpu  internal, called by full.

Run (-B: do not write __pycache__ into any env):
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -B -X utf8 scripts\\swarm2\\verify_E1_1_probe.py full
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -B -X utf8 scripts\\swarm2\\verify_E1_1_probe.py tiny
"""

import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import xgboost as xgb

SEED = 2026  # ASSUMPTION: independent of the E1 seed (12345)
N_FEAT = 20  # ASSUMPTION
DEPTH = 4  # same as the claimed benchmark
ETA = 0.1  # ASSUMPTION (same as the claimed benchmark)


def make(n_train, n_hold, seed=SEED):
    rng = np.random.default_rng(seed)
    n = n_train + n_hold
    X = rng.standard_normal((n, N_FEAT), dtype=np.float32)
    logit = (
        1.0 * X[:, 0]
        - 0.7 * X[:, 1]
        + 0.6 * X[:, 2] * X[:, 3]
        + 0.5 * np.cos(X[:, 4])
        + 0.8 * (X[:, 5] > 0.5)
        + 0.7 * rng.standard_normal(n)
    ).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    yy = np.asarray(y, dtype=np.float64)
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


def find_key(obj, key):
    """First string value stored under `key`, searched recursively in the saved config."""
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], str):
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def build_info():
    print("sys.executable", sys.executable)
    print("xgboost", xgb.__version__, xgb.__file__)
    try:
        print("shared library", xgb.core._LIB._name)
    except Exception as e:  # noqa: BLE001
        print("shared library path unavailable:", type(e).__name__, str(e)[:200])
    bi = xgb.build_info()
    for k in ("USE_CUDA", "CUDA_VERSION", "USE_NCCL", "USE_OPENMP", "libxgboost"):
        print("build_info", k, "=", bi.get(k))


def train_device(dev, X_tr, y_tr, X_ho, y_ho, n_trees):
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dho = xgb.DMatrix(X_ho)
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": dev,
        "max_depth": DEPTH,
        "eta": ETA,
        "seed": 0,
    }
    t0 = time.perf_counter()
    b = xgb.train(params, dtr, num_boost_round=n_trees)
    dt = time.perf_counter() - t0
    device_seen = find_key(json.loads(b.save_config()), "device")
    p = b.predict(dho)
    return b, dt, device_seen, p


def smi(query):
    r = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.stdout.strip() or r.stderr.strip()


def smi_apps():
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return (r.stdout.strip() or r.stderr.strip()) or "(empty)"


def mode_full():
    build_info()
    X_tr, y_tr, X_ho, y_ho = make(1_000_000, 50_000)
    base = logloss(np.full(len(y_ho), y_tr.mean()), y_ho)
    print(f"data: train {X_tr.shape} holdout {X_ho.shape} base_rate_holdout_logloss={base:.6f}")

    before = smi("memory.used,utilization.gpu")
    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append(smi("memory.used,utilization.gpu"))
            time.sleep(0.5)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    b, dt, device_seen, p = train_device("cuda", X_tr, y_tr, X_ho, y_ho, 50)
    stop.set()
    th.join(timeout=10)
    after = smi("memory.used,utilization.gpu")

    print(
        f"cuda train: rounds={b.num_boosted_rounds()} time={dt:.3f}s saved_config_device={device_seen} "
        f"holdout_logloss={logloss(p, y_ho):.6f}"
    )
    print("gpu mem MiB, util % (memory.used, utilization.gpu) before:", before)
    print("gpu during train (sampled every 0.5 s):", samples)
    print("gpu after:", after)
    print("compute apps (pid, used_memory MiB):", smi_apps())
    print("this process pid:", os.getpid())

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    r = subprocess.run(
        [sys.executable, "-I", "-B", "-X", "utf8", os.path.abspath(__file__), "child_nogpu"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    print("control child (CUDA_VISIBLE_DEVICES=-1) returncode:", r.returncode)
    print(r.stdout.strip())
    if r.stderr.strip():
        print("control child stderr (tail):", r.stderr.strip()[-600:])


def mode_child():
    build_info()
    X_tr, y_tr, X_ho, y_ho = make(20_000, 5_000)
    try:
        b, dt, device_seen, p = train_device("cuda", X_tr, y_tr, X_ho, y_ho, 5)
        print(f"child cuda OK: saved_config_device={device_seen} rounds={b.num_boosted_rounds()}")
    except Exception as e:  # noqa: BLE001
        print("child cuda FAILED:", type(e).__name__, str(e)[:600])


def mode_tiny():
    build_info()
    X_tr, y_tr, X_ho, y_ho = make(20_000, 5_000)
    for dev in ("cpu", "cuda"):
        try:
            b, dt, device_seen, p = train_device(dev, X_tr, y_tr, X_ho, y_ho, 5)
            print(f"device {dev} OK: saved_config_device={device_seen} rounds={b.num_boosted_rounds()}")
        except Exception as e:  # noqa: BLE001
            print(f"device {dev} FAILED:", type(e).__name__, str(e)[:600])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "full":
        mode_full()
    elif mode == "tiny":
        mode_tiny()
    elif mode == "child_nogpu":
        mode_child()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
