"""verify_E1_2 (real-feature part): does the CUDA vs CPU gap show up on the project's own CS2 features?

Data: data/viz/per_tick_states.parquet (1,962,000 per-tick rows, 1,521 matches, 28,683 rounds: the
bake-off kept set). Features and monotone constraints are copied from scripts/model/bakeoff.py (FEATS, MONO).
Split: rebuilt with the same logic as bakeoff.make_splits (SEED 0, TEST_FRAC 0.20, VAL_FRAC 0.10, sorted keys,
random.Random(SEED).shuffle). The rebuilt split is checked against data/viz/split.json (sets train 1095 /
val 122 / test 304 matches; rows 1,395,102 / 162,065 / 404,833; and the 3 listed test matches).
Training uses the primary train set only (no early stopping, fixed number of trees; ASSUMPTION).
Holdout = primary test set.

Two parameter sets (both ASSUMPTION about the exact bake-off settings, see the notes in the log):
  R1 "E1 params on real features": hist, depth 4, eta 0.1, 200 trees, no subsample, no monotone constraints.
  R2 "bake-off-like": hist, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, monotone constraints,
     300 trees (the bake-off uses early stopping up to 3000 trees; not reproduced), seed 0.

Modes (run from the repo root):
  cpu_g : run with the G: project interpreter (CPU-only xgboost). Saves CPU predictions to the scratchpad.
    G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_E1_2_real.py cpu_g
  cuda  : run with the cs2gpu interpreter (CUDA xgboost). Trains CPU (nthread=1) and CUDA, the references,
          and compares its CPU predictions with the cpu_g predictions if those files exist.
    C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_E1_2_real.py cuda

Nothing is written under E:. Intermediate arrays go to the scratchpad only.
"""
import json
import os
import random
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

TABLE = os.path.join("data", "viz", "per_tick_states.parquet")
SPLIT_FILE = os.path.join("data", "viz", "split.json")
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"

# copied from scripts/model/bakeoff.py
FEATS = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
         "seconds_since_plant", "round_number", "ct_score_minus_t_score", "is_pistol_round",
         "is_overtime"]
MONO_MAP = {"alive_ct": 1, "alive_t": -1, "ct_alive_minus_t": 1, "ct_score_minus_t_score": 1,
            "seconds_since_plant": 0}
MONO = [MONO_MAP.get(f, 0) for f in FEATS]
SEED = 0
TEST_FRAC = 0.20
VAL_FRAC = 0.10
LABEL = "label_ct_win"

CONFIGS = {
    "R1_E1params": {"n_trees": 200,
                    "params": {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 4,
                               "eta": 0.1, "seed": 0}},
    "R2_bakeoff_like": {"n_trees": 300,
                        "params": {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 4,
                                   "eta": 0.05, "min_child_weight": 20, "subsample": 0.8,
                                   "monotone_constraints": "(" + ",".join(str(m) for m in MONO) + ")",
                                   "seed": 0}},
}


def rebuild_split(keys):
    keys = sorted(keys)
    shuffled = list(keys)
    random.Random(SEED).shuffle(shuffled)
    n_test = round(TEST_FRAC * len(keys))
    test, pool = shuffled[:n_test], shuffled[n_test:]
    n_val = round(VAL_FRAC * len(pool))
    val, train = pool[:n_val], pool[n_val:]
    return {"train": set(train), "val": set(val), "test": set(test)}


def load_data():
    df = pl.read_parquet(TABLE, columns=["match_key"] + FEATS + [LABEL])
    keys = df["match_key"].unique().to_list()
    sp = rebuild_split(keys)
    split_ref = json.load(open(SPLIT_FILE, encoding="utf-8"))
    counts = {}
    for name in ("train", "val", "test"):
        m = df.filter(pl.col("match_key").is_in(list(sp[name])))
        counts[name] = {"matches": len(sp[name]), "rows": m.height}
    ref = split_ref["sets"]
    checks = {
        "matches_match_split_json": all(counts[k]["matches"] == ref[k]["matches"] for k in counts),
        "rows_match_split_json": all(counts[k]["rows"] == ref[k]["rows"] for k in counts),
        "listed_test_matches_in_rebuilt_test": all(t in sp["test"] for t in split_ref["test_matches"]),
    }
    print("[split] rebuilt counts:", json.dumps(counts), flush=True)
    print("[split] split.json counts:", json.dumps({k: {"matches": ref[k]["matches"], "rows": ref[k]["rows"]}
                                                    for k in ref}), flush=True)
    print("[split] checks:", json.dumps(checks), flush=True)
    if not all(checks.values()):
        raise SystemExit("rebuilt split does not match split.json; stop")
    mk = df["match_key"]
    tr = df.filter(mk.is_in(list(sp["train"])))
    te = df.filter(mk.is_in(list(sp["test"])))
    X_tr = tr.select(FEATS).to_numpy().astype(np.float32)
    y_tr = tr[LABEL].to_numpy().astype(np.float32)
    X_te = te.select(FEATS).to_numpy().astype(np.float32)
    y_te = te[LABEL].to_numpy().astype(np.float32)
    print(f"[data] train {X_tr.shape} ct_win_rate={y_tr.mean():.6f}; test {X_te.shape} ct_win_rate={y_te.mean():.6f}",
          flush=True)
    return X_tr, y_tr, X_te, y_te, checks


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
    la, lb = per_row_logloss(p_a, y), per_row_logloss(p_b, y)
    d = lb - la
    se = float(d.std(ddof=1) / np.sqrt(d.size))
    return {"gap_b_minus_a": float(d.mean()), "paired_se": se,
            "z": float(d.mean() / se) if se > 0 else float("nan")}


def tree_nodes(dump_json):
    out = []
    for t in dump_json:
        seq = []

        def rec(node):
            if "leaf" in node:
                seq.append(("leaf", float(node["leaf"])))
            else:
                seq.append(("split", str(node["split"]), float(node["split_condition"])))
                for ch in node["children"]:
                    rec(ch)

        rec(json.loads(t))
        out.append(seq)
    return out


def structure_report(b_a, b_b, la, lb):
    ta = tree_nodes(b_a.get_dump(dump_format="json"))
    tb = tree_nodes(b_b.get_dump(dump_format="json"))
    n_same, first_diff, leaf_max = 0, None, []
    for i, (sa, sb) in enumerate(zip(ta, tb)):
        sa_s = [(x[0], x[1], x[2]) if x[0] == "split" else ("leaf",) for x in sa]
        sb_s = [(x[0], x[1], x[2]) if x[0] == "split" else ("leaf",) for x in sb]
        if sa_s == sb_s:
            n_same += 1
            va = [x[1] for x in sa if x[0] == "leaf"]
            vb = [x[1] for x in sb if x[0] == "leaf"]
            leaf_max.append(float(np.max(np.abs(np.array(va) - np.array(vb)))) if va else 0.0)
        elif first_diff is None:
            first_diff = i
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
    return {"n_trees": len(ta), "n_trees_same_structure": n_same, "first_structure_diff_tree": first_diff,
            "leaf_max_abs_diff_same_structure_max": float(max(leaf_max)) if leaf_max else None,
            "distinct_thresholds_shared": sum(len(thr_a.get(f, set()) & thr_b.get(f, set())) for f in feats),
            f"distinct_thresholds_only_{la}": sum(len(thr_a.get(f, set()) - thr_b.get(f, set())) for f in feats),
            f"distinct_thresholds_only_{lb}": sum(len(thr_b.get(f, set()) - thr_a.get(f, set())) for f in feats)}


def cut_arrays(d):
    ptr, val = d.get_quantile_cut()
    return np.asarray(ptr, dtype=np.int64), np.asarray(val, dtype=np.float64)


def cut_report(ca, cb):
    pa, va = ca
    pb, vb = cb
    rep = {"total_cuts_a": int(va.size), "total_cuts_b": int(vb.size), "ptr_identical": bool(np.array_equal(pa, pb))}
    if pa.shape == pb.shape and va.shape == vb.shape:
        fin = np.isfinite(va) & np.isfinite(vb)
        dif = np.abs(va[fin] - vb[fin])
        rep.update({"finite_cuts_compared": int(fin.sum()),
                    "finite_cut_max_abs_diff": float(dif.max()) if dif.size else 0.0,
                    "finite_cuts_identical": int(np.sum(dif == 0.0))})
    else:
        rep["finite_cut_max_abs_diff"] = "shape mismatch"
    return rep


def train(params, n_trees, X, y, device, nthread=None):
    """Train on a fresh DMatrix. Returns (booster, dmatrix, seconds). The DMatrix keeps the cuts of this device."""
    p = dict(params)
    p["device"] = device
    if nthread is not None:
        p["nthread"] = nthread
    d = xgb.DMatrix(X, label=y)
    t0 = time.perf_counter()
    b = xgb.train(p, d, num_boost_round=n_trees)
    dt = time.perf_counter() - t0
    return b, d, dt


def cmd_cpu_g(X_tr, y_tr, X_te, y_te):
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)
    dte = xgb.DMatrix(X_te)
    for name, cfg in CONFIGS.items():
        b, _, dt = train(cfg["params"], cfg["n_trees"], X_tr, y_tr, "cpu", nthread=1)
        p = b.predict(dte)
        out = os.path.join(SCRATCH, f"verify_E1_2_real_cpu_g_{name}.npy")
        np.save(out, p)
        print(f"[cpu_g] {name} trained {dt:.1f}s logloss={logloss(p, y_te):.6f} saved={out}", flush=True)


def cmd_cuda(X_tr, y_tr, X_te, y_te):
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)
    dte = xgb.DMatrix(X_te)
    res = {}
    for name, cfg in CONFIGS.items():
        nt = cfg["n_trees"]
        P = cfg["params"]
        print(f"==== {name}: {json.dumps(P)} trees={nt} ====", flush=True)
        b_cpu, d_cpu, t_cpu = train(P, nt, X_tr, y_tr, "cpu", nthread=1)
        p_cpu = b_cpu.predict(dte)
        cut_cpu = cut_arrays(d_cpu)
        b_cuda, d_cuda, t_cuda = train(P, nt, X_tr, y_tr, "cuda")
        p_cuda = b_cuda.predict(dte)
        cut_cuda = cut_arrays(d_cuda)
        print(f"[{name}] cpu nthread=1 {t_cpu:.1f}s; cuda {t_cuda:.1f}s", flush=True)

        # Cross-device prediction on the same two models
        b_cuda.set_param({"device": "cpu"})
        p_cuda_on_cpu = b_cuda.predict(dte)
        b_cpu.set_param({"device": "cuda"})
        p_cpu_on_gpu = b_cpu.predict(dte)

        # References: materiality of ordinary small changes on CPU
        rng = np.random.default_rng(1)
        keep = rng.random(len(y_tr)) < 0.99
        if name == "R1_E1params":
            ref_label = "cpu_99pct_rows"
            b_ref, _, _ = train(P, nt, X_tr[keep], y_tr[keep], "cpu", nthread=1)
        else:
            ref_label = "cpu_seed1_same_rows"
            P_ref = dict(P, seed=1)
            b_ref, _, _ = train(P_ref, nt, X_tr, y_tr, "cpu", nthread=1)
        p_ref = b_ref.predict(dte)

        r = {"logloss": {"cpu": logloss(p_cpu, y_te), "cuda": logloss(p_cuda, y_te),
                         "cuda_model_on_cpu_predictor": logloss(p_cuda_on_cpu, y_te),
                         "cpu_model_on_gpu_predictor": logloss(p_cpu_on_gpu, y_te),
                         ref_label: logloss(p_ref, y_te)},
             "diff_cpu_vs_cuda": diff_summary(p_cpu, p_cuda),
             "gap_cuda_minus_cpu": paired_gap(p_cpu, p_cuda, y_te),
             "diff_cuda_model_cpu_pred_vs_native": diff_summary(p_cuda_on_cpu, p_cuda),
             "diff_cpu_model_gpu_pred_vs_native": diff_summary(p_cpu_on_gpu, p_cpu),
             "diff_cpu_vs_" + ref_label: diff_summary(p_cpu, p_ref),
             "gap_" + ref_label + "_minus_cpu": paired_gap(p_cpu, p_ref, y_te),
             "diff_cuda_vs_" + ref_label: diff_summary(p_cuda, p_ref),
             "gap_" + ref_label + "_minus_cuda": paired_gap(p_cuda, p_ref, y_te),
             "structure_cpu_vs_cuda": structure_report(b_cpu, b_cuda, "cpu", "cuda"),
             "cuts_cpu_vs_cuda": cut_report(cut_cpu, cut_cuda)}
        # Saved CPU predictions from the G: env (cross-build check)
        g_path = os.path.join(SCRATCH, f"verify_E1_2_real_cpu_g_{name}.npy")
        if os.path.exists(g_path):
            p_g = np.load(g_path)
            r["cpu_build_check_G_vs_C"] = diff_summary(p_g, p_cpu)
        res[name] = r
        print(f"[{name}] logloss {json.dumps({k: round(v, 6) for k, v in r['logloss'].items()})}", flush=True)
        print(f"[{name}] diff cpu vs cuda {json.dumps(r['diff_cpu_vs_cuda'])}", flush=True)
        print(f"[{name}] gap cuda-cpu {json.dumps(r['gap_cuda_minus_cpu'])}", flush=True)
        print(f"[{name}] cuda model, CPU predictor vs GPU predictor {json.dumps(r['diff_cuda_model_cpu_pred_vs_native'])}", flush=True)
        print(f"[{name}] cpu model, GPU predictor vs CPU predictor {json.dumps(r['diff_cpu_model_gpu_pred_vs_native'])}", flush=True)
        print(f"[{name}] reference {ref_label}: diff vs cpu {json.dumps(r['diff_cpu_vs_' + ref_label])}", flush=True)
        print(f"[{name}] reference {ref_label}: gap vs cpu {json.dumps(r['gap_' + ref_label + '_minus_cpu'])}", flush=True)
        print(f"[{name}] reference {ref_label}: diff vs cuda {json.dumps(r['diff_cuda_vs_' + ref_label])}", flush=True)
        print(f"[{name}] structure {json.dumps(r['structure_cpu_vs_cuda'])}", flush=True)
        print(f"[{name}] cuts {json.dumps(r['cuts_cpu_vs_cuda'])}", flush=True)
        if "cpu_build_check_G_vs_C" in r:
            print(f"[{name}] CPU build check G env vs C env {json.dumps(r['cpu_build_check_G_vs_C'])}", flush=True)
    print("[summary-json] " + json.dumps(res), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    X_tr, y_tr, X_te, y_te, checks = load_data()
    if mode == "cpu_g":
        cmd_cpu_g(X_tr, y_tr, X_te, y_te)
    elif mode == "cuda":
        cmd_cuda(X_tr, y_tr, X_te, y_te)
    else:
        raise SystemExit("mode must be cpu_g or cuda")
