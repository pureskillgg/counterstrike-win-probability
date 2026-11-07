"""verify_E1_1 parity: replicate the E1 CPU-vs-CUDA numbers and test the strongest alternative.

Context: the E1-1 claim says "device=cuda trains". This script checks that the CUDA model is
a real, repeatable model (determinism) and characterizes how it differs from CPU, because the
E1 notes say the two are not numerically equivalent (E1-2). Same generator and settings as
scripts\\swarm2\\E1_parity.py (seed 12345, 50 float32 N(0,1) features, 200 trees, depth 4,
eta 0.1, hist), at 100k train and 100k holdout.

Steps:
  1. Replicate: cpu (nthread=1) vs cuda holdout logloss and prediction differences.
  2. Determinism: train cpu twice and cuda twice; exact equality expected for a repeatable run.
  3. First divergence: first boosting round where the tree split features or split
     thresholds differ between cpu and cuda; per-round holdout logloss gap.

Run (-B: do not write __pycache__ into any env):
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -B -X utf8 scripts\\swarm2\\verify_E1_1_parity.py
"""

import json
import time

import numpy as np
import xgboost as xgb

SEED = 12345  # same as E1_parity.py
N_FEAT = 50
N_HOLD = 100_000
N_TREES = 200
MAX_DEPTH = 4
ETA = 0.1
N_TRAIN = 100_000


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


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    yy = np.asarray(y, dtype=np.float64)
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


def train(dev, dtr, dho, yho):
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": dev,
        "max_depth": MAX_DEPTH,
        "eta": ETA,
        "seed": 0,
    }
    if dev == "cpu":
        params["nthread"] = 1
    res = {}
    t0 = time.perf_counter()
    b = xgb.train(
        params,
        dtr,
        num_boost_round=N_TREES,
        evals=[(dho, "ho")],
        evals_result=res,
        verbose_eval=False,
    )
    dt = time.perf_counter() - t0
    p = b.predict(dho)
    return b, p, dt, res["ho"]["logloss"]


def walk(node, feats, conds, leaves):
    if "leaf" in node:
        leaves.append(float(node["leaf"]))
        return
    feats.append(node["split"])
    conds.append(float(node["split_condition"]))
    for c in node["children"]:
        walk(c, feats, conds, leaves)


def tree_structs(b):
    out = []
    for s in b.get_dump(dump_format="json"):
        t = json.loads(s)
        feats, conds, leaves = [], [], []
        walk(t, feats, conds, leaves)
        out.append((feats, conds, leaves))
    return out


def main():
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)
    Xtr, ytr, Xho, yho = make_data(N_TRAIN)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dho = xgb.DMatrix(Xho, label=yho)
    dho_nolabel = xgb.DMatrix(Xho)

    runs = {}
    for name, dev in (("cpu_a", "cpu"), ("cpu_b", "cpu"), ("cuda_a", "cuda"), ("cuda_b", "cuda")):
        b, p, dt, curve = train(dev, dtr, dho, yho)
        runs[name] = {"booster": b, "pred": np.asarray(p, dtype=np.float64), "time": dt, "curve": curve}
        print(f"{name:<7} device={dev:<4} time={dt:8.3f}s holdout_logloss={logloss(p, yho):.6f}", flush=True)

    print("\n-- determinism (exact equality of predictions) --")
    print("cpu_a == cpu_b:", np.array_equal(runs["cpu_a"]["pred"], runs["cpu_b"]["pred"]),
          "max_abs", float(np.max(np.abs(runs["cpu_a"]["pred"] - runs["cpu_b"]["pred"]))))
    print("cuda_a == cuda_b:", np.array_equal(runs["cuda_a"]["pred"], runs["cuda_b"]["pred"]),
          "max_abs", float(np.max(np.abs(runs["cuda_a"]["pred"] - runs["cuda_b"]["pred"]))))

    pc = runs["cpu_a"]["pred"]
    pg = runs["cuda_a"]["pred"]
    d = np.abs(pc - pg)
    print("\n-- cpu vs cuda (same as E1_parity 100k; E1 reported max 0.1997, mean 0.0186, gap +0.000258) --")
    print(f"max_abs={d.max():.6f} mean_abs={d.mean():.6f} frac_abs_gt_0.01={np.mean(d > 0.01):.4f} "
          f"logloss_gap(cuda-cpu)={logloss(pg, yho) - logloss(pc, yho):+.6f}")

    print("\n-- per-round holdout logloss (cuda minus cpu) --")
    cc = np.asarray(runs["cpu_a"]["curve"])
    gg = np.asarray(runs["cuda_a"]["curve"])
    diff = gg - cc
    for r in (1, 2, 5, 10, 25, 50, 100, 150, 200):
        print(f"round {r:>3}: cpu={cc[r - 1]:.6f} cuda={gg[r - 1]:.6f} diff={diff[r - 1]:+.6f}")
    print(f"max |diff| over rounds {np.max(np.abs(diff)):.6f}")

    print("\n-- first divergence in tree structure (cpu_a vs cuda_a) --")
    sc = tree_structs(runs["cpu_a"]["booster"])
    sg = tree_structs(runs["cuda_a"]["booster"])
    first_feat = next((i for i, (a, c) in enumerate(zip(sc, sg)) if a[0] != c[0]), None)
    first_cond = next((i for i, (a, c) in enumerate(zip(sc, sg)) if a[1] != c[1]), None)
    first_leaf = next((i for i, (a, c) in enumerate(zip(sc, sg))
                       if len(a[2]) != len(c[2]) or np.max(np.abs(np.array(a[2]) - np.array(c[2]))) > 1e-6), None)
    print(f"trees compared: {len(sc)} vs {len(sg)}")
    print(f"first tree (0-based) with different split features: {first_feat}")
    print(f"first tree with different split thresholds (exact float): {first_cond}")
    print(f"first tree with leaf values differing by > 1e-6: {first_leaf}")
    if first_cond is not None:
        i = first_cond
        print(f"tree {i} cpu features {sc[i][0][:8]} ... cuda features {sg[i][0][:8]} ...")
        print(f"tree {i} cpu thresholds {[round(x, 5) for x in sc[i][1][:6]]}")
        print(f"tree {i} cuda thresholds {[round(x, 5) for x in sg[i][1][:6]]}")
    print(f"tree 0 max |leaf diff| = {np.max(np.abs(np.array(sc[0][2]) - np.array(sg[0][2]))):.6g}")


if __name__ == "__main__":
    main()
