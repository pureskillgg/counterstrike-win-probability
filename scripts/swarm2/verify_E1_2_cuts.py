"""verify_E1_2 (mechanism probe): per-feature quantile cut counts, CPU vs CUDA, on the real CS2 features and on the
quantized synthetic features. Question: does the CUDA sketch keep fewer candidate thresholds than the CPU sketch
(especially for low-cardinality features)?

Interpreter (needs CUDA): C:\\Users\\13144\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_E1_2_cuts.py
Run from the repo root (reads data/viz/per_tick_states.parquet and data/viz/split.json; reads nothing under E:).
Each DMatrix is trained for one tree (cuts are built during training), so this is cheap.
"""
import json
import random

import numpy as np
import polars as pl
import xgboost as xgb

FEATS = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
         "seconds_since_plant", "round_number", "ct_score_minus_t_score", "is_pistol_round",
         "is_overtime"]
SEED, TEST_FRAC, VAL_FRAC = 0, 0.20, 0.10
LABEL = "label_ct_win"


def real_train():
    df = pl.read_parquet("data/viz/per_tick_states.parquet", columns=["match_key"] + FEATS + [LABEL])
    keys = sorted(df["match_key"].unique().to_list())
    shuffled = list(keys)
    random.Random(SEED).shuffle(shuffled)
    n_test = round(TEST_FRAC * len(keys))
    test, pool = shuffled[:n_test], shuffled[n_test:]
    n_val = round(VAL_FRAC * len(pool))
    train = set(pool[n_val:])
    tr = df.filter(pl.col("match_key").is_in(list(train)))
    return tr.select(FEATS).to_numpy().astype(np.float32), tr[LABEL].to_numpy().astype(np.float32), FEATS


def synth_quantized(n_train=100_000, n_hold=100_000):
    # Same generator, seed and draw order as verify_E1_2_synth.make_data (copied, not imported).
    rng = np.random.default_rng(12345)
    n = n_train + n_hold
    X = rng.standard_normal((n, 50), dtype=np.float32)
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
    Xq = (np.round(X[:n_train] * 4) / 4).astype(np.float32)
    return Xq, y[:n_train], [f"s{j}" for j in range(50)]


def cut_counts(X, y, device):
    d = xgb.DMatrix(X, label=y)
    xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": device, "max_depth": 4,
               "eta": 0.1, "seed": 0}, d, num_boost_round=1)
    ptr, val = d.get_quantile_cut()
    ptr = np.asarray(ptr)
    val = np.asarray(val, dtype=np.float64)
    out = []
    for j in range(X.shape[1]):
        vals = val[ptr[j]:ptr[j + 1]]
        fin = vals[np.isfinite(vals)]
        out.append(fin)
    return out


def probe(label, X, y, names):
    uniq = [len(np.unique(X[:, j])) for j in range(X.shape[1])]
    cpu = cut_counts(X, y, "cpu")
    cuda = cut_counts(X, y, "cuda")
    rows = []
    for j in range(X.shape[1]):
        data_vals = np.unique(X[:, j]).astype(np.float64)
        c_cpu, c_cuda = cpu[j], cuda[j]
        rows.append({"feature": names[j], "distinct_values": uniq[j],
                     "finite_cuts_cpu": int(c_cpu.size), "finite_cuts_cuda": int(c_cuda.size),
                     "cpu_cuts_equal_a_data_value": int(np.isin(c_cpu, data_vals).sum()),
                     "cuda_cuts_equal_a_data_value": int(np.isin(c_cuda, data_vals).sum())})
    tot = {"features": len(rows), "finite_cuts_cpu": sum(r["finite_cuts_cpu"] for r in rows),
           "finite_cuts_cuda": sum(r["finite_cuts_cuda"] for r in rows),
           "features_cuda_fewer_cuts": sum(1 for r in rows if r["finite_cuts_cuda"] < r["finite_cuts_cpu"]),
           "features_cuda_more_cuts": sum(1 for r in rows if r["finite_cuts_cuda"] > r["finite_cuts_cpu"]),
           "features_equal_count": sum(1 for r in rows if r["finite_cuts_cuda"] == r["finite_cuts_cpu"])}
    print(f"==== {label} ====", flush=True)
    for r in rows:
        print(json.dumps(r), flush=True)
    print("[totals]", json.dumps(tot), flush=True)


if __name__ == "__main__":
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)
    Xr, yr, names_r = real_train()
    probe("REAL per-tick CS2 train features (1.395M rows, 10 features)", Xr, yr, names_r)
    Xs, ys, names_s = synth_quantized()
    probe("SYNTHETIC quantized features (100k rows, 50 features, round to 0.25)", Xs, ys, names_s)
