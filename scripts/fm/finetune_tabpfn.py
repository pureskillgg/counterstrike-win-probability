"""Does FINE-TUNING a pretrained tabular foundation model (TabPFN v2) beat the tuned XGBoost on this task?

  G:\\pureskill\\envs\\tabicl\\Scripts\\python.exe -X utf8 scripts\\fm\\finetune_tabpfn.py [--version v2] [--epochs 10] [--device cuda]

The zero-shot head-to-head (TabICL v2, Mitra, in-context only) lost by 0.009-0.013 log loss. The user's point: a foundation model needs fine-tuning or special
training to win. This script runs three arms on the SAME held-out test seconds as the XGBoost reference in data/<store>/fm/test.parquet (written by prep_fm_data.py):
  zero-shot   the pretrained TabPFN (chosen model version), a context of --ctx-rows seconds from training matches, no weight updates
  fine-tuned  FinetunedTabPFNClassifier: the weights are updated on meta-datasets (context + query) sampled from TRAINING matches (at most --per-round seconds per round),
              early stopping on log loss of an EXPLICIT validation set drawn from the VALIDATION matches (never random rows: seconds of one round are near-duplicates,
              so a row-level validation split would leak and stop too late)
  reference   XGBoost, already scored in test.parquet (p_xgb): the paired difference carries a match-cluster 95% interval
Versions: v2 weights are under the Prior Labs License v1.1 (Apache 2.0 + attribution, commercial use allowed); v2.5 and later are non-commercial and are refused here
unless --allow-noncommercial is given (research use only).
Writes data/<store>/exp_finetune.json after each arm.
"""
import os
import re

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import json
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def rows_per_round(G, keys, feats, per_round, seed, cap=None):
    rng = np.random.default_rng(seed)
    sub = G.filter(pl.col("match_key").is_in(sorted(keys)))
    sub = sub.with_columns(pl.Series("_r", rng.random(sub.height))).sort("_r")
    sub = sub.with_columns(pl.int_range(pl.len()).over(["match_key", "round"]).alias("_k")).filter(pl.col("_k") < per_round)
    if cap:
        sub = sub.head(cap)
    return sub.select(feats).to_numpy().astype(np.float32), sub["label_ct_win"].to_numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per-round", type=int, default=3)
    ap.add_argument("--ctx-rows", type=int, default=20000, help="context rows for zero-shot and for final inference of the fine-tuned model")
    ap.add_argument("--ctx-query", type=int, default=16000, help="context + query rows per fine-tuning meta-dataset")
    ap.add_argument("--val-rows", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=0, help="first n test seconds only (0 = all 20,000)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--arms", default="zero,finetune")
    ap.add_argument("--allow-noncommercial", action="store_true")
    a = ap.parse_args()
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier

    ver = {v.value: v for v in ModelVersion}[a.version]
    if a.version != "v2" and not a.allow_noncommercial:
        sys.exit(f"TabPFN {a.version} weights are non-commercial; pass --allow-noncommercial for research use only")
    base = os.path.join(W.ROOT, "data", a.store)
    meta = json.load(open(os.path.join(base, "fm", "meta.json"), encoding="utf-8"))
    feats = meta["features"]
    d = np.load(os.path.join(base, "fm", "ft_data.npz"))  # written by prep_ft_data.py (training / validation matches only)
    Xtr_all, ytr_all, Xva, yva = d["Xtr"], d["ytr"], d["Xva"][:a.val_rows], d["yva"][:a.val_rows]
    te = pl.read_parquet(os.path.join(base, "fm", "test.parquet"))
    if a.n_test:
        te = te.head(a.n_test)
    Xt = te.select(feats).to_numpy().astype(np.float32)
    yt = te["label_ct_win"].to_numpy().astype(np.int64)
    p_ref = te["p_xgb"].to_numpy()
    sub = te.select("match_key", "w_round")
    r_ref = W.evaluate(sub, yt, p_ref)
    print(f"train rows {len(ytr_all):,} (<= {a.per_round}/round, training matches only), validation rows {len(yva):,} (validation matches), test {len(yt):,} seconds from "
          f"{te['match_key'].n_unique()} held-out matches; XGBoost LL {r_ref['logloss']:.4f} {[round(x, 4) for x in r_ref['logloss_ci95']]} AUC {r_ref['auc']:.4f}", flush=True)
    out_path = os.path.join(base, "exp_finetune.json")
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}

    def record(name, p, fit_s, pred_s, extra):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        r = W.evaluate(sub, yt, p, ref=p_ref)
        np.save(os.path.join(base, 'fm', 'pred_' + re.sub(r'[^A-Za-z0-9]+', '_', name) + '.npy'), p)  # kept so arms can be compared PAIRED with each other later
        res[name] = {"version": a.version, "test_seconds": int(len(yt)), "logloss": r["logloss"], "ci95": r["logloss_ci95"], "auc": r["auc"], "ece10": r["ece10"],
                     "smooth_ece": r["smooth_ece"], "delta_vs_xgb": r["delta_vs_ref"], "delta_ci95": r["delta_vs_ref_ci95"], "xgb_logloss": r_ref["logloss"],
                     "xgb_ci95": r_ref["logloss_ci95"], "fit_seconds": fit_s, "predict_seconds": pred_s, "rows_per_second": len(yt) / max(pred_s, 1e-9), **extra}
        W.dump_json(res, out_path)
        print(f"{name}: LL {r['logloss']:.4f} {[round(x, 4) for x in r['logloss_ci95']]} AUC {r['auc']:.4f} ECE {r['ece10']:.4f} | delta vs XGBoost {r['delta_vs_ref']:+.4f} "
              f"{[round(x, 4) for x in r['delta_vs_ref_ci95']]} | fit {fit_s:.0f}s predict {pred_s:.0f}s", flush=True)

    arms = a.arms.split(",")
    if "zero" in arms:
        t = time.time()
        clf = TabPFNClassifier.create_default_for_version(ver, device=a.device)
        clf.fit(Xtr_all[:a.ctx_rows], ytr_all[:a.ctx_rows])
        fit_s = time.time() - t
        t = time.time()
        p = clf.predict_proba(Xt)[:, 1]
        record(f"tabpfn-{a.version} zero-shot ctx{a.ctx_rows}", p, fit_s, time.time() - t, {"arm": "zero-shot", "context_rows": int(min(a.ctx_rows, len(ytr_all)))})
        del clf
    if "finetune" in arms:
        import torch
        t = time.time()
        ft = FinetunedTabPFNClassifier(device=a.device, epochs=a.epochs, learning_rate=a.lr, eval_metric="log_loss", model_version=ver,
                                       n_finetune_ctx_plus_query_samples=a.ctx_query, n_inference_subsample_samples=a.ctx_rows, early_stopping=True,
                                       early_stopping_patience=a.patience, validation_split_ratio=None)
        ft.fit(Xtr_all, ytr_all, Xva, yva)
        fit_s = time.time() - t
        t = time.time()
        p = ft.predict_proba(Xt)[:, 1]
        record(f"tabpfn-{a.version} fine-tuned ep{a.epochs} lr{a.lr:g} ctx{a.ctx_rows}", p, fit_s, time.time() - t,
               {"arm": "fine-tuned", "epochs_max": a.epochs, "learning_rate": a.lr, "train_rows": int(len(ytr_all)), "val_rows": int(len(yva)),
                "context_rows": int(a.ctx_rows), "gpu_peak_gb": (torch.cuda.max_memory_allocated() / 1e9) if a.device == "cuda" else None})


if __name__ == "__main__":
    main()
