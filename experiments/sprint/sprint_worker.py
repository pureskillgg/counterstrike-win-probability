"""Validation sprint worker: fits the cells of a queue on one device, reading the matrix that sprint_prepare.py wrote.

A cell is {id, set, depth, eta, mcw, seed}. The matrix holds the training-cache rows that have a train-centre position row, ordered
fit, early stop, validation; holdout B and the frozen test are not in it (sprint_prepare.py asserts this). Each fit reads two
contiguous slices of the matrix, so the worker does not hold the whole design in memory.

    CS2_ROOT=/mnt/g/pureskill/csds-cs2 python sprint_worker.py --queue <queue.json> --matrix <matrix dir> --out <run dir> --device cuda

A cell whose results/<id>.json has status ok is skipped, so a rerun resumes. Each cell writes preds/<id>.parquet (match_key, round,
second, label_ct_win, p) and results/<id>.json. A cell that raises writes its traceback with status error, and the queue goes on.
The run folder's manifest.json records the matrix and settings; a folder built from other settings is refused.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("CS2_ROOT", "G:/pureskill/csds-cs2")
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

import wp_common as W  # noqa: E402

# the fit path of this worker; a run folder records it in its manifest, so results from another path are refused, not mixed
FIT_PATH = "xgb.train on a QuantileDMatrix quantised once per feature set (max_bin 256); early stopping 100 on the early-stop set; 3000-round cap"

TRAIN_CACHE = os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet")
HOLDOUT_B_KEYS = os.path.join(ROOT, "data", "combined", "holdout_22_25_test_keys.json")
GEO = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_nearest_site_mean", "t_nearest_site_mean", "c_spread", "t_spread",
       "ct_t_centroid_dist", "min_enemy_dist", "c_in_contact", "t_in_contact", "c_min_bomb_dist", "c_n_bomb_500", "t_min_bomb_dist",
       "defuse_slack"]
VIEW = ["c_sees_enemy", "t_sees_enemy", "c_exposed", "t_exposed"]
BLIND = ["c_blind_n", "t_blind_n", "c_blind_sum", "t_blind_sum"]
GROUPS = {"GEO": GEO, "VIEW": VIEW, "BLIND": BLIND}
POS_KEY = ["match_key", "round", "tick"]
POS_NAME = "cache_pos_2026_trainctr.parquet"  # site centres fitted on training matches only; the all-match table is refused
ES_SHARE = 10  # percent of the training matches held out for early stopping
SETS = {  # "base" is the 59 cache inputs; the others add position groups
    "base": ["base"],
    "full": ["base", "GEO", "VIEW", "BLIND"],
    "base+GEO": ["base", "GEO"],
    "base+VIEW": ["base", "VIEW"],
    "base+BLIND": ["base", "BLIND"],
    "base+VIEW+BLIND": ["base", "VIEW", "BLIND"],
    "base+GEO+BLIND": ["base", "GEO", "BLIND"],
    "base+GEO+VIEW": ["base", "GEO", "VIEW"],
}


def early_stop_set(keys):
    return {k for k in keys if zlib.crc32(f"es:{k}".encode()) % 100 < ES_SHARE}


def rss_mb():
    """Resident memory of this process in MB (Linux, including WSL); None elsewhere. Logged so a memory kill shows its phase."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def cell_features(set_name, base):
    feats = []
    for g in SETS[set_name]:
        feats += base if g == "base" else GROUPS[g]
    return feats


def load_matrix(mdir):
    meta = json.load(open(os.path.join(mdir, "meta.json")))
    X = np.load(os.path.join(mdir, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(mdir, "y.npy"), mmap_mode="r")
    assert X.shape == (meta["rows"], len(meta["feats_all"])), "matrix shape does not match its meta.json"
    return meta, X, y, pl.read_parquet(os.path.join(mdir, "val_keys.parquet"))


def fit_cell(data, c, device, pred_dir):
    feats = cell_features(c["set"], data["base"])
    idx = [data["feats_all"].index(f) for f in feats]
    n_fit, n_es = data["n_fit"], data["n_es"]
    a1, a2 = n_fit, n_fit + n_es
    X, y = data["X"], data["y"]
    Xtr, ytr = np.ascontiguousarray(X[:a1][:, idx]), np.asarray(y[:a1])
    Xes, yes = np.ascontiguousarray(X[a1:a2][:, idx]), np.asarray(y[a1:a2])
    print(f"{c['id']}: matrices read, rss {rss_mb()} MB; fitting", flush=True)
    t0 = time.time()
    m, best = W.fit_xgb(Xtr, ytr, Xes, yes, W.mono_for(feats), seed=c["seed"], device=device,
                        depth=c["depth"], eta=c["eta"], mcw=c["mcw"])
    fit_s = time.time() - t0
    print(f"{c['id']}: fit done in {fit_s:.0f}s, best {best}, rss {rss_mb()} MB", flush=True)
    del Xtr, Xes
    gc.collect()
    Xva, yv = np.ascontiguousarray(X[a2:][:, idx]), np.asarray(y[a2:])
    p = W.predict(m, best, Xva)
    ll = float(-(yv * np.log(np.clip(p, 1e-15, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-15, 1))).mean())
    pred_path = os.path.join(pred_dir, c["id"] + ".parquet")
    data["val_keys"].with_columns(pl.Series("p", p)).write_parquet(pred_path)
    return {"best_iteration": int(best), "val_logloss": ll, "fit_seconds": round(fit_s, 1), "n_features": len(feats),
            "n_fit": int(n_fit), "n_early_stop": int(n_es), "n_validation": int(len(yv)), "pred_file": pred_path}


def prepare_set(data, set_name):
    """Quantise the fit, early-stop and validation rows once for a feature set. Every cell of that set reuses these matrices."""
    feats = cell_features(set_name, data["base"])
    idx = [data["feats_all"].index(f) for f in feats]
    a1, a2 = data["n_fit"], data["n_fit"] + data["n_es"]
    X, y = data["X"], data["y"]
    t0 = time.time()
    dtrain = xgb.QuantileDMatrix(np.ascontiguousarray(X[:a1][:, idx]), np.asarray(y[:a1]), max_bin=256)
    dval = xgb.QuantileDMatrix(np.ascontiguousarray(X[a1:a2][:, idx]), np.asarray(y[a1:a2]), ref=dtrain, max_bin=256)
    dva = xgb.QuantileDMatrix(np.ascontiguousarray(X[a2:][:, idx]), np.asarray(y[a2:]), ref=dtrain, max_bin=256)
    build_s = round(time.time() - t0, 1)
    print(f"{set_name}: quantised {len(feats)} features in {build_s}s, rss {rss_mb()} MB", flush=True)
    return {"set": set_name, "feats": feats, "dtrain": dtrain, "dval": dval, "dva": dva, "yv": np.asarray(y[a2:]),
            "build_s": build_s}


def fit_cell_prepared(prep, c, device, pred_dir, data):
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": c["eta"], "max_depth": c["depth"],
              "min_child_weight": c["mcw"], "subsample": 0.8, "tree_method": "hist", "device": device, "seed": c["seed"],
              "monotone_constraints": tuple(W.mono_for(prep["feats"]))}
    t0 = time.time()
    bst = xgb.train(params, prep["dtrain"], num_boost_round=3000, evals=[(prep["dval"], "validation_0")],
                    early_stopping_rounds=100, verbose_eval=False)
    fit_s = time.time() - t0
    best = int(bst.best_iteration)
    print(f"{c['id']}: fit done in {fit_s:.0f}s, best {best}, rss {rss_mb()} MB", flush=True)
    p = bst.predict(prep["dva"], iteration_range=(0, best + 1))
    yv = prep["yv"]
    ll = float(-(yv * np.log(np.clip(p, 1e-15, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-15, 1))).mean())
    pred_path = os.path.join(pred_dir, c["id"] + ".parquet")
    data["val_keys"].with_columns(pl.Series("p", p)).write_parquet(pred_path)
    return {"best_iteration": best, "val_logloss": ll, "fit_seconds": round(fit_s, 1), "quantise_seconds": prep["build_s"],
            "n_features": len(prep["feats"]), "n_fit": int(data["n_fit"]), "n_early_stop": int(data["n_es"]),
            "n_validation": int(len(yv)), "pred_file": pred_path, "fit_path": FIT_PATH}


def write_json(path, obj):
    """Write to a temp file, fsync, then rename, so a kill leaves the old file or the new one, never a partial one."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def done(res_dir, cell_id):
    path = os.path.join(res_dir, cell_id + ".json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
    except Exception:
        return False  # an unreadable file of any kind is a cell that did not finish: it is refitted
    return isinstance(r, dict) and r.get("status") == "ok"


def definitions_fingerprint(meta):
    """Everything that changes what a cell means: the feature groups and sets, the matrix's feature list, the monotone constraints,
    the split, and the source of the fit and predict functions in wp_common. A change to any of these changes the fingerprint."""
    parts = {"SETS": SETS, "GROUPS": GROUPS, "feats_all": meta["feats_all"], "base": meta["base"], "MONO_ALIVE": W.MONO_ALIVE,
             "fit_xgb": inspect.getsource(W.fit_xgb), "predict": inspect.getsource(W.predict), "mono_for": inspect.getsource(W.mono_for),
             "ES_SHARE": ES_SHARE, "split": "make_split random seed 0"}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def run_manifest(meta, device):
    """What a run folder was built from. A folder built from other settings or another device is refused, not mixed."""
    return {"device": device, "definitions": definitions_fingerprint(meta), "pos": meta["pos"], "pos_bytes": meta["pos_bytes"],
            "cache": meta["cache"], "cache_bytes": meta["cache_bytes"], "frozen": meta["frozen"], "rows": meta["rows"],
            "mono_extra": os.environ.get("WP_EXTRA_MONO", "0"), "split_env": os.environ.get("WP_SPLIT_MANIFEST", "default")}


def check_manifest(out_dir, manifest):
    path = os.path.join(out_dir, "manifest.json")
    res_dir = os.path.join(out_dir, "results")
    has_results = os.path.isdir(res_dir) and any(f.endswith(".json") for f in os.listdir(res_dir))
    if not os.path.exists(path):
        if has_results:
            raise SystemExit(f"{out_dir} holds results but no manifest.json; refusing to adopt them. Use a new run folder.")
        write_json(path, manifest)
        return
    old = json.load(open(path))
    changed = sorted(k for k in manifest if old.get(k) != manifest[k])
    if changed:
        raise SystemExit(f"{out_dir} holds results from other settings or another device (changed: {', '.join(changed)}); "
                         "use a new run folder")


def main():
    ap = argparse.ArgumentParser(description="Validation sprint worker")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--matrix", required=True, help="the folder sprint_prepare.py wrote")
    ap.add_argument("--out", required=True, help="this device's run folder")
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    a = ap.parse_args()
    meta, X, y, val_keys = load_matrix(a.matrix)
    if meta["pos"] != POS_NAME:
        raise SystemExit(f"the matrix was built from {meta['pos']}, not {POS_NAME}")
    cells = json.load(open(a.queue))
    res_dir, pred_dir = os.path.join(a.out, "results"), os.path.join(a.out, "preds")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    check_manifest(a.out, dict(run_manifest(meta, a.device), fit_path=FIT_PATH))
    todo = [c for c in cells if not done(res_dir, c["id"])]
    print(f"queue {len(cells)} cells, {len(todo)} to run on {a.device}", flush=True)
    if not todo:
        return
    data = {"X": X, "y": y, "feats_all": meta["feats_all"], "base": meta["base"], "n_fit": meta["n_fit"], "n_es": meta["n_es"],
            "val_keys": val_keys}
    print(f"matrix {X.shape}: fit {meta['n_fit']:,}, early stop {meta['n_es']:,}, validation {meta['n_val']:,}; rss {rss_mb()} MB",
          flush=True)
    # cells are grouped by feature set, in the order their sets first appear, so each set is quantised once
    by_set = {}
    for c in todo:
        by_set.setdefault(c["set"], []).append(c)
    groups = list(by_set.items())
    # the next set is quantised on the CPU while the current set trains on the GPU; the results do not depend on this overlap
    pool = ThreadPoolExecutor(max_workers=1)
    pending = pool.submit(prepare_set, data, groups[0][0]) if groups else None
    for i, (set_name, group) in enumerate(groups):
        prep, prep_err = None, None
        try:
            prep = pending.result()
        except Exception:
            prep_err = traceback.format_exc()
        pending = pool.submit(prepare_set, data, groups[i + 1][0]) if i + 1 < len(groups) else None
        for c in group:
            started = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                if prep is None:
                    raise RuntimeError(f"quantising {set_name} failed:\n{prep_err}")
                r = fit_cell_prepared(prep, c, a.device, pred_dir, data)
                r["status"] = "ok"
            except Exception:
                r = {"status": "error", "traceback": traceback.format_exc()}
            r.update(c)
            r.update({"device": a.device, "pos": meta["pos"], "started": started,
                      "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
            write_json(os.path.join(res_dir, c["id"] + ".json"), r)
            print(f"{c['id']}: {r['status']} val_logloss {r.get('val_logloss')} best {r.get('best_iteration')} "
                  f"fit {r.get('fit_seconds')}s", flush=True)
        del prep
        gc.collect()


if __name__ == "__main__":
    main()
