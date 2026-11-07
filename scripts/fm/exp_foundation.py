"""Do pretrained tabular foundation models (in-context learners) match tuned XGBoost on the round win-probability task?

First (project CPU environment):   scripts\\fm\\prep_fm_data.py      -> data/<store>/fm/{context.npz, test.parquet, meta.json}
Then, inside an isolated environment (setup_fm_envs.ps1):
  G:\\pureskill\\envs\\tabicl\\Scripts\\python.exe -X utf8 scripts\\fm\\exp_foundation.py --model tabicl --ctx 10000 --device cpu
  G:\\pureskill\\envs\\mitra\\Scripts\\python.exe  -X utf8 scripts\\fm\\exp_foundation.py --model mitra  --ctx 10000 --device cpu

The foundation model is given a CONTEXT of the first `--ctx` rows of context.npz (labelled seconds from TRAINING matches only, one per round: prep_fm_data.py's cap of 2 kept 1 because cum_count was 1-based; fixed since) and
predicts P(CT wins) for the held-out test seconds in test.parquet. test.parquet also holds the final XGBoost settings' prediction for the same seconds,
so the paired log-loss difference (with a match-cluster interval) is the honest comparison. Reported: log loss, AUC, ECE, paired delta, fit time,
predict time and the prediction rate in rows per second on the chosen device (a live service needs about one row per game second per match).
Writes data/<store>/exp_foundation.json after every run, keyed by model, context size, device and fine-tune flag.
"""
import os

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


def run_tabicl(Xc, yc, Xt, device, args):
    from tabicl import TabICLClassifier
    t = time.time()
    # device is always explicit: the default (None) would silently pick the GPU, and the question is CPU speed
    clf = TabICLClassifier(device=device, n_estimators=args.n_estimators, kv_cache=bool(args.kv_cache), random_state=0, verbose=False)
    clf.fit(Xc, yc)
    fit_s = time.time() - t
    t = time.time()
    p = clf.predict_proba(Xt)[:, 1]
    return p, fit_s, time.time() - t


def run_mitra(Xc, yc, Xt, device, args):
    import pandas as pd
    from autogluon.tabular import TabularPredictor
    cols = [f"f{i}" for i in range(Xc.shape[1])]
    tr = pd.DataFrame(Xc, columns=cols)
    tr["y"] = yc
    te = pd.DataFrame(Xt, columns=cols)
    t = time.time()
    pred = TabularPredictor(label="y", eval_metric="log_loss", verbosity=1, path=os.path.join(os.environ.get("TEMP", "."), "ag_mitra_run")).fit(
        tr, hyperparameters={"MITRA": {"fine_tune": bool(args.fine_tune), "ag.max_memory_usage_ratio": args.mem_ratio}}, num_gpus=1 if device == "cuda" else 0, time_limit=args.time_limit)
    fit_s = time.time() - t
    t = time.time()
    p = pred.predict_proba(te).iloc[:, -1].to_numpy()
    return p, fit_s, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--model", choices=["tabicl", "mitra"], required=True)
    ap.add_argument("--ctx", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=0, help="use only the first n test seconds (0 = all)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fine-tune", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=8, help="TabICL ensemble size (its default is 8; fewer is faster)")
    ap.add_argument("--kv-cache", action="store_true", help="TabICL: cache the context once so each later prediction only pays for the query rows")
    ap.add_argument("--time-limit", type=int, default=3600)
    ap.add_argument("--mem-ratio", type=float, default=1.0, help="AutoGluon refuses a model whose memory estimate exceeds this share of free RAM (Mitra at 5,000 context rows is estimated at 18.6 GB); raise it only with RAM to spare")
    args = ap.parse_args()
    base = os.path.join(W.ROOT, "data", args.store)
    meta = json.load(open(os.path.join(base, "fm", "meta.json"), encoding="utf-8"))
    ctx = np.load(os.path.join(base, "fm", "context.npz"))
    Xc, yc = ctx["X"][:args.ctx], ctx["y"][:args.ctx]
    te = pl.read_parquet(os.path.join(base, "fm", "test.parquet"))
    if args.n_test:
        te = te.head(args.n_test)
    Xt = te.select(meta["features"]).to_numpy().astype(np.float32)
    yt = te["label_ct_win"].to_numpy().astype(np.int64)
    p_ref = te["p_xgb"].to_numpy()
    print(f"context {len(yc):,} rows (CT-win rate {yc.mean():.3f}); test {len(yt):,} seconds from {te['match_key'].n_unique()} held-out matches", flush=True)
    r_ref = W.evaluate(te.select("match_key", "w_round"), yt, p_ref)
    print(f"XGBoost on these seconds: LL {r_ref['logloss']:.4f} {[round(x, 4) for x in r_ref['logloss_ci95']]} AUC {r_ref['auc']:.4f}", flush=True)

    runner = {"tabicl": run_tabicl, "mitra": run_mitra}[args.model]
    p, fit_s, pred_s = runner(Xc, yc, Xt, args.device, args)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    r = W.evaluate(te.select("match_key", "w_round"), yt, p, ref=p_ref)
    name = f"{args.model}|ctx{len(yc)}|{args.device}{'|ft' if args.fine_tune else ''}" + (f"|est{args.n_estimators}" if args.model == "tabicl" else "") + ("|kv" if args.kv_cache else "") + f"|n{len(yt)}"
    out_path = os.path.join(base, "exp_foundation.json")
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    res[name] = {"model": args.model, "context_rows": int(len(yc)), "device": args.device, "fine_tune": bool(args.fine_tune), "test_seconds": int(len(yt)),
                 "logloss": r["logloss"], "ci95": r["logloss_ci95"], "auc": r["auc"], "ece10": r["ece10"], "smooth_ece": r["smooth_ece"],
                 "delta_vs_xgb": r["delta_vs_ref"], "delta_ci95": r["delta_vs_ref_ci95"], "xgb_logloss": r_ref["logloss"], "xgb_ci95": r_ref["logloss_ci95"],
                 "fit_seconds": fit_s, "predict_seconds": pred_s, "rows_per_second": len(yt) / max(pred_s, 1e-9)}
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"{name}: LL {r['logloss']:.4f} {[round(x, 4) for x in r['logloss_ci95']]} AUC {r['auc']:.4f} ECE {r['ece10']:.4f} | delta vs XGBoost {r['delta_vs_ref']:+.4f} "
          f"{[round(x, 4) for x in r['delta_vs_ref_ci95']]} | fit {fit_s:.0f}s predict {pred_s:.0f}s = {len(yt) / max(pred_s, 1e-9):.0f} rows/s", flush=True)


if __name__ == "__main__":
    main()
