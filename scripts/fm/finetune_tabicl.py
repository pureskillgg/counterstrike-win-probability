"""Fine-tune TabICL v2 (BSD-3-Clause code and checkpoint: commercial use allowed) and score it paired against the tuned XGBoost.

  G:\\pureskill\\envs\\tabicl\\Scripts\\python.exe -X utf8 scripts\\fm\\finetune_tabicl.py [--epochs 6] [--device cuda]

Same protocol as finetune_tabpfn.py (read its docstring): training rows come from TRAINING matches only (at most --per-round seconds per round), early stopping uses an
EXPLICIT validation set from the VALIDATION matches (log loss; the library's default is ROC-AUC, which ignores calibration), and the arms are scored on the first --n-test
seconds of data/<store>/fm/test.parquet, where the XGBoost reference prediction (p_xgb) was stored, so every difference is paired with a match-cluster 95% interval.
Arms: zero-shot (pretrained checkpoint, context = first --ctx-rows training seconds) and fine-tuned. Writes data/<store>/exp_finetune.json.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per-round", type=int, default=3)
    ap.add_argument("--ctx-rows", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=8000, help="max_data_size: rows per fine-tuning meta-dataset (context + query)")
    ap.add_argument("--val-rows", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--max-train", type=int, default=20000, help="cap on training rows: TabICL's fine-tuning code held 14.8 GB of memory on 47,574 rows (24 GB on 143k), so it is trained on a random subset")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--n-estimators", type=int, default=8)
    ap.add_argument("--arms", default="zero,finetune")
    a = ap.parse_args()
    from tabicl import FinetunedTabICLClassifier, TabICLClassifier

    base = os.path.join(W.ROOT, "data", a.store)
    meta = json.load(open(os.path.join(base, "fm", "meta.json"), encoding="utf-8"))
    feats = meta["features"]
    d = np.load(os.path.join(base, "fm", "ft_data.npz"))  # written by prep_ft_data.py (training / validation matches only)
    Xtr, ytr, Xva, yva = d["Xtr"][:a.max_train], d["ytr"][:a.max_train], d["Xva"][:a.val_rows], d["yva"][:a.val_rows]
    te = pl.read_parquet(os.path.join(base, "fm", "test.parquet")).head(a.n_test)
    Xt = te.select(feats).to_numpy().astype(np.float32)
    yt = te["label_ct_win"].to_numpy().astype(np.int64)
    p_ref = te["p_xgb"].to_numpy()
    sub = te.select("match_key", "w_round")
    r_ref = W.evaluate(sub, yt, p_ref)
    print(f"train rows {len(ytr):,} (<= {a.per_round}/round, training matches only), validation rows {len(yva):,} (validation matches), test {len(yt):,} seconds from "
          f"{te['match_key'].n_unique()} held-out matches; XGBoost LL {r_ref['logloss']:.4f} {[round(x, 4) for x in r_ref['logloss_ci95']]} AUC {r_ref['auc']:.4f}", flush=True)
    out_path = os.path.join(base, "exp_finetune_tabicl.json")  # its own file: the TabPFN script writes exp_finetune.json (two runs at once must not overwrite each other)
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}

    def record(name, p, fit_s, pred_s, extra):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        r = W.evaluate(sub, yt, p, ref=p_ref)
        np.save(os.path.join(base, 'fm', 'pred_' + re.sub(r'[^A-Za-z0-9]+', '_', name) + '.npy'), p)  # kept so arms can be compared PAIRED with each other later
        res[name] = {"version": "tabicl-v2", "test_seconds": int(len(yt)), "logloss": r["logloss"], "ci95": r["logloss_ci95"], "auc": r["auc"], "ece10": r["ece10"],
                     "smooth_ece": r["smooth_ece"], "delta_vs_xgb": r["delta_vs_ref"], "delta_ci95": r["delta_vs_ref_ci95"], "xgb_logloss": r_ref["logloss"],
                     "xgb_ci95": r_ref["logloss_ci95"], "fit_seconds": fit_s, "predict_seconds": pred_s, "rows_per_second": len(yt) / max(pred_s, 1e-9), **extra}
        W.dump_json(res, out_path)
        print(f"{name}: LL {r['logloss']:.4f} {[round(x, 4) for x in r['logloss_ci95']]} AUC {r['auc']:.4f} ECE {r['ece10']:.4f} | delta vs XGBoost {r['delta_vs_ref']:+.4f} "
              f"{[round(x, 4) for x in r['delta_vs_ref_ci95']]} | fit {fit_s:.0f}s predict {pred_s:.0f}s", flush=True)

    arms = a.arms.split(",")
    if "zero" in arms:
        t = time.time()
        clf = TabICLClassifier(device=a.device, n_estimators=a.n_estimators, random_state=0)
        clf.fit(Xtr[:a.ctx_rows], ytr[:a.ctx_rows])
        fit_s = time.time() - t
        t = time.time()
        p = clf.predict_proba(Xt)[:, 1]
        record(f"tabicl-v2 zero-shot ctx{a.ctx_rows}", p, fit_s, time.time() - t, {"arm": "zero-shot", "context_rows": int(a.ctx_rows)})
        del clf
    if "finetune" in arms:
        import torch
        t = time.time()
        ft = FinetunedTabICLClassifier(device=a.device, epochs=a.epochs, learning_rate=a.lr, eval_metric="log_loss", max_data_size=a.chunk, n_estimators_inference=a.n_estimators,
                                       early_stopping=True, patience=a.patience, verbose=True, random_state=0)
        ft.fit(Xtr, ytr, X_val=Xva, y_val=yva)
        fit_s = time.time() - t
        t = time.time()
        p = ft.predict_proba(Xt)[:, 1]
        record(f"tabicl-v2 fine-tuned ep{a.epochs} lr{a.lr:g} chunk{a.chunk}", p, fit_s, time.time() - t,
               {"arm": "fine-tuned", "epochs_max": a.epochs, "learning_rate": a.lr, "train_rows": int(len(ytr)), "val_rows": int(len(yva)), "chunk": a.chunk,
                "gpu_peak_gb": (torch.cuda.max_memory_allocated() / 1e9) if a.device == "cuda" else None})


if __name__ == "__main__":
    main()
