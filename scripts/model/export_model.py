"""Train the deployable round win-probability model, save it for CPU inference, and run the GPU-vs-CPU and calibration checks.

  ...cs2gpu python.exe -I -X utf8 scripts\\model\\export_model.py --store combined --sets hp,map --name winprob_v1

Outputs  models/<name>.json (XGBoost booster, loadable on CPU), models/<name>.meta.json, data/<store>/export_<name>.json
Checks written to the report:
  * GPU-trained booster loaded on CPU predicts the same numbers (max |diff|) as it did on GPU
  * a model trained on CPU with the same settings: test log loss difference with a match-cluster interval
  * isotonic and Platt recalibration fitted on validation only: test log loss vs the raw probabilities
  * reliability table with match-cluster intervals on the held-out test matches
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import hashlib
import json
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import features_events as FE
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]


def pav_isotonic(p_fit, y_fit):
    """Pool-adjacent-violators on (p, y) sorted by p; returns a function mapping new p to the fitted step values."""
    o = np.argsort(p_fit, kind="mergesort")
    x, y = p_fit[o], y_fit[o].astype(np.float64)
    # merge equal x first
    ux, start, cnt = np.unique(x, return_index=True, return_counts=True)
    sy = np.add.reduceat(y, start)
    w = cnt.astype(np.float64)
    means = sy / w
    # PAV with a stack
    vals, wts, ends = [], [], []
    for i in range(len(ux)):
        vals.append(means[i]); wts.append(w[i]); ends.append(i)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            tv = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw
            vals[-2:] = [tv]; wts[-2:] = [tw]; ends[-2:] = [ends[-1]]
    ends = np.array(ends)
    fitted = np.empty(len(ux))
    s = 0
    for v, e in zip(vals, ends):
        fitted[s:e + 1] = v
        s = e + 1
    return lambda q: np.interp(q, ux, fitted)


def platt(p_fit, y_fit):
    z = np.log(np.clip(p_fit, 1e-6, 1 - 1e-6) / (1 - np.clip(p_fit, 1e-6, 1 - 1e-6)))
    A = np.stack([np.ones_like(z), z], 1)
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-(A @ w)))
        g = A.T @ (p - y_fit)
        H = (A * (p * (1 - p))[:, None]).T @ A + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return lambda q: 1 / (1 + np.exp(-(w[0] + w[1] * np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--tomes", default=None)
    ap.add_argument("--sets", default="")
    ap.add_argument("--name", default="winprob_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eras", default="")
    ap.add_argument("--stride", type=int, default=1, help="train/validate on every stride-th second (evaluation always uses all)")
    ap.add_argument("--split", default="random")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="device of the main fit (cpu: the GPU-vs-CPU comparison is then skipped)")
    ap.add_argument("--from-cache", action="store_true", help="read data/<store>/cache_snapshot_2026.parquet (make_snapshot_cache.py) instead of recomputing the snapshot features")
    args = ap.parse_args()
    tomes = args.tomes or args.store
    t0 = time.time()
    sets = [s for s in args.sets.split(",") if s]
    if args.from_cache:
        # the snapshot feature table built once by make_snapshot_cache.py: exactly the rows and columns this script would build for
        # --sets map,status --eras 2026 (roster-defect and table-vs-events rounds left out, map one-hots and the loadout features attached),
        # without holding the grid, the raw features and the copies at once (the 19 M-row grid made that need ~15 GB)
        if sorted(sets) != ["map", "status"] or args.eras != "2026":
            raise SystemExit("--from-cache only reproduces --sets map,status --eras 2026")
        G = pl.read_parquet(os.path.join(W.ROOT, "data", args.store, "cache_snapshot_2026.parquet"))
        G = G.drop([c for c in ("label_source", "upload_day", "build_num", "score_lead_diag") if c in G.columns])
        feats = list(W.F6) + [f"map_is_{m}" for m in MAPS] + [c for c in G.columns if c.startswith("st_")]
        print(f"loaded the snapshot cache: {G.height:,} rows, {G['match_key'].n_unique():,} matches", flush=True)
    else:
        # read only the columns the fit and the report use: the four dropped below include two long string columns (upload_day, label_source) that,
        # at 19 M rows, cost about 1.3 GB of host memory for nothing
        names = pl.scan_parquet(os.path.join(W.ROOT, "data", args.store, "grid.parquet")).collect_schema().names()
        G = W.load_grid(args.store, cols=[c for c in names if c not in ("label_source", "upload_day", "build_num", "score_lead_diag")])
        if args.eras:
            G = G.filter(pl.col("era").is_in(args.eras.split(",")))
        G = G.drop([c for c in ("label_source", "upload_day", "build_num", "score_lead_diag") if c in G.columns])
        if "status" in sets:
            import features_status as FS
            n0 = G.height
            G = W.drop_roster_defect_rounds(G, args.store)
            n1 = G.height
            G = FS.drop_inconsistent_rounds(G)  # rounds where the raw player table and the event-derived alive counts disagree at any tick
            print(f"roster-defect rounds left out: {n0 - n1} grid rows ({(n0 - n1) / n0:.2%}); table-vs-events inconsistent rounds left out: {n1 - G.height} rows", flush=True)
        feats = list(W.F6)
        if "hp" in sets:
            G = G.with_columns(FE.hp_features(G, tomes))
            feats += ["hp_ct", "hp_t"]
        if "map" in sets:
            G = G.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Int8).alias(f"map_is_{m}") for m in MAPS])
            feats += [f"map_is_{m}" for m in MAPS]
        if "status" in sets:  # living-player loadout snapshot (2026 only): the serving model derives the same numbers from a player table
            import features_status as FS
            new = FS.loadout_features(G)
            G = G.with_columns(new)
            feats += new.columns
        for mod in [s for s in sets if s.startswith("feat_")]:
            import importlib
            new = importlib.import_module(mod).compute(G, tomes)
            G = G.with_columns(new)
            feats += new.columns
    print("features:", len(feats), feats, flush=True)
    sp = W.make_split(args.split, G, seed=args.seed)
    te_sub, Xte, yte = W.arrays(G, sp["test"], feats)
    _, Xtr, ytr = W.arrays(G, sp["train"], feats, stride=args.stride)
    va_sub, Xva, yva = W.arrays(G, sp["val"], feats, stride=args.stride)
    mono = W.mono_for(feats)
    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=args.seed, device=args.device)
    p_gpu = W.predict(m, best, Xte)
    pv_gpu = W.predict(m, best, Xva)
    os.makedirs(os.path.join(W.ROOT, "models"), exist_ok=True)
    path = os.path.join(W.ROOT, "models", args.name + ".json")
    m.get_booster().save_model(path)

    # --- CPU inference of the GPU-trained booster
    import xgboost as xgb
    b = xgb.Booster()
    b.set_param({"device": "cpu"})
    b.load_model(path)
    p_cpu_load = b.predict(xgb.DMatrix(Xte), iteration_range=(0, best + 1))
    report = {"features": feats, "best_iteration": best,
              "cpu_inference_of_gpu_model_max_abs_diff": float(np.max(np.abs(p_cpu_load - p_gpu)))}

    # --- CPU-trained model, same settings
    mc, bestc = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=args.seed, device="cpu")
    p_cpu = W.predict(mc, bestc, Xte)
    r_gpu = W.evaluate(te_sub, yte, p_gpu)
    r_cpu = W.evaluate(te_sub, yte, p_cpu, ref=p_gpu)
    KEYS = ("logloss", "logloss_ci95", "logloss_round_weighted", "brier", "brier_ci95", "auc", "ece10", "ece10_ci95", "ece20_equal_count", "smooth_ece")
    report["main_fit_device"] = args.device
    report["gpu_trained"] = {k: r_gpu[k] for k in KEYS}  # the main fit (the name is historical: with --device cpu this is the CPU fit)
    report["cpu_trained"] = {k: r_cpu[k] for k in KEYS}
    report["cpu_minus_gpu_logloss"] = {"delta": r_cpu["delta_vs_ref"], "ci95": r_cpu["delta_vs_ref_ci95"], "cpu_best_iteration": bestc}

    # --- recalibration, fitted on validation only
    iso = pav_isotonic(pv_gpu, yva)
    pl_ = platt(pv_gpu, yva.astype(np.float64))
    for name, fn in (("isotonic", iso), ("platt", pl_)):
        r = W.evaluate(te_sub, yte, np.clip(fn(p_gpu), 1e-6, 1 - 1e-6), ref=p_gpu)
        report[name] = {"logloss": r["logloss"], "delta_vs_raw": r["delta_vs_ref"], "delta_ci95": r["delta_vs_ref_ci95"], "ece10": r["ece10"],
                        "ece10_ci95": r["ece10_ci95"], "ece20_equal_count": r["ece20_equal_count"], "smooth_ece": r["smooth_ece"]}
    report["raw_calibration"] = r_gpu["calibration"]
    report["test"] = {"matches": r_gpu["n_matches"], "rows": r_gpu["n_rows"]}
    for era in sorted(te_sub["era"].unique().to_list()):
        msk = (te_sub["era"] == era).to_numpy()
        r = W.evaluate(te_sub.filter(pl.Series(msk)), yte[msk], p_gpu[msk])
        report[f"test_era_{era}"] = {k: r[k] for k in ("n_matches", "logloss", "logloss_ci95", "logloss_round_weighted", "auc", "ece10", "ece10_ci95", "smooth_ece")}
    # --- every tick: log loss and calibration by seconds since the round went live, and by bomb phase / alive state
    secs = te_sub["seconds_since_round_live"].to_numpy()
    planted = te_sub["bomb_planted"].to_numpy().astype(bool)
    ac_, at_ = te_sub["alive_ct"].to_numpy(), te_sub["alive_t"].to_numpy()
    groups = {f"t_{lo}-{hi}s_preplant": (~planted) & (secs >= lo) & (secs < hi) for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90), (90, 105), (105, 400))}
    groups.update({f"plant_{lo}-{hi}s_after": planted & (te_sub["seconds_since_plant"].to_numpy() >= lo) & (te_sub["seconds_since_plant"].to_numpy() < hi)
                   for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 100))})
    groups.update({"5v5": (ac_ == 5) & (at_ == 5), "even_not_5v5": (ac_ == at_) & (ac_ < 5), "ct_ahead": ac_ > at_, "t_ahead": at_ > ac_,
                   "last_alive_1v1_or_less": (ac_ <= 1) & (at_ <= 1), "pistol": te_sub["is_pistol_round"].to_numpy().astype(bool),
                   "overtime": te_sub["is_overtime"].to_numpy().astype(bool)})
    for mp in sorted(te_sub["map_name"].unique().to_list()):
        groups["map_" + str(mp)] = (te_sub["map_name"] == mp).to_numpy()
    report["by_group"] = {}
    for name, msk in groups.items():
        if msk.sum() >= 500:
            r = W.evaluate(te_sub.filter(pl.Series(msk)), yte[msk], p_gpu[msk])
            report["by_group"][name] = {"rows": int(msk.sum()), "matches": r["n_matches"], "logloss": r["logloss"], "logloss_ci95": r["logloss_ci95"],
                                        "mean_pred": float(p_gpu[msk].mean()), "observed": float(yte[msk].mean()), "ece10": r["ece10"]}
    report["train"] = {"matches": len(sp["train"]), "val_matches": len(sp["val"]), "rows": int(len(ytr)), "stride": args.stride, "split": args.split,
                       "eras": dict(G.filter(pl.col("match_key").is_in(sorted(sp["train"]))).group_by("era").len().iter_rows())}
    def sha256_file(p):  # the WHOLE file, in 1 MiB blocks (the earlier version hashed only the first block)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    d_store = os.path.join(W.ROOT, "data", args.store)
    maps_in_data = sorted(str(x) for x in G["map_name"].unique().to_list())  # scope is what the model was fitted on, read from the data
    man = W.load_manifest()
    meta = {"version": args.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "features": feats, "best_iteration": best,
            "monotone_constraints": dict(zip(feats, mono)), "prior_ct_win": float(ytr.mean()),
            "scope": {"maps": maps_in_data, "maps_expected": ["de_" + m for m in MAPS], "modes": ["competitive"],
                      "excluded": ["wingman", "hostage", "other maps"]},
            "training": report["train"], "xgboost": xgb.__version__, "seed": args.seed, "main_fit_device": args.device,
            "grid_file_sha256": sha256_file(os.path.join(d_store, "grid.parquet")),
            "matches_file_sha256": sha256_file(os.path.join(d_store, "matches.parquet")),
            "rounds_file_sha256": sha256_file(os.path.join(d_store, "rounds.parquet")),
            "split_manifest_sha256": man["sha256"] if man else None, "split": args.split,
            "test_logloss": r_gpu["logloss"], "test_logloss_ci95": r_gpu["logloss_ci95"]}
    json.dump(meta, open(os.path.join(W.ROOT, "models", args.name + ".meta.json"), "w", encoding="utf-8"), indent=1)
    out = os.path.join(W.ROOT, "data", args.store, f"export_{args.name}.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1)
    # row sample for the end-to-end stress test: grid rows of the held-out matches with the pipeline predictions
    # every second of every held-out round with the model's probability: the input of inspect_rounds.py (looking at single rounds is leak-free
    # because these matches were never used to fit or to stop the model)
    keep = [c for c in ("match_key", "round", "tick", "second", "seconds_since_round_live", "alive_ct", "alive_t", "bomb_planted", "seconds_since_plant",
                        "label_ct_win", "map_name", "era", "round_number") if c in te_sub.columns]
    te_sub.select(keep).with_columns(pl.Series("p", p_gpu)).write_parquet(os.path.join(W.ROOT, "data", args.store, f"export_{args.name}_test_predictions.parquet"))
    samp = te_sub.with_columns(pl.Series("p_pipeline", p_gpu)).sample(n=min(20000, te_sub.height), seed=0)
    samp.write_parquet(os.path.join(W.ROOT, "data", args.store, f"export_{args.name}_test_sample.parquet"))
    print(json.dumps({k: report[k] for k in ("cpu_inference_of_gpu_model_max_abs_diff", "gpu_trained", "cpu_trained", "cpu_minus_gpu_logloss", "isotonic", "platt")}, indent=1))
    print(f"-> {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
