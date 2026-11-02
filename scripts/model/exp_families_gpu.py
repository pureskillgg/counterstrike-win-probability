"""GPU versions of the comparison families that have one, on the same frozen split and the same scoring as exp_full_comparison.py.

  catboost   CatBoost with task_type GPU, the CPU run's settings (depth 6, lr 0.08, early stop 100 on validation, monotone constraints).
             If the GPU refuses the monotone constraints, the fit is repeated without them and the row says so.
  mlp        PyTorch on GPU. Same architecture as the sklearn MLP (128-64, ReLU), Adam lr 1e-3, batch 4096, 10 epochs, a 25% row sample,
             standardised inputs. sklearn's L2 penalty (alpha 1e-4) is not replicated, so this is not a bit-for-bit copy.
  cuml_logit cuML LogisticRegression (C 1.0, max_iter 300) on standardised inputs.
  cuml_rf    cuML RandomForestClassifier, 100 trees, min leaf 50, 10% row sample. cuML uses quantile-binned splits and a default
             max depth of 16, so it is a different algorithm from sklearn's random forest.

XGBoost is reloaded from models/ (its GPU fit), as in exp_full_comparison.py, so every delta is against the same reference.
Writes data/<store>/<out>; combine the parts with merge_comparison.py.

  ...python.exe -X utf8 scripts/model/exp_families_gpu.py [--models catboost,mlp,cuml_logit,cuml_rf] [--xgb-model winprob_v4] [--out exp_families_gpu.json]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def to_np(x):
    """cuML and cupy return device arrays; bring them to the host as numpy."""
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--models", default="catboost,mlp,cuml_logit,cuml_rf")
    ap.add_argument("--xgb-model", default="winprob_v4")
    ap.add_argument("--sub-rf", type=float, default=0.1)
    ap.add_argument("--sub-mlp", type=float, default=0.25)
    ap.add_argument("--out", default="exp_families_gpu.json")
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    names = [n for n in a.models.split(",") if n]
    t0 = time.time()

    # the same data path as exp_full_comparison.py (streamed float32 load)
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats))
    lf = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    G = lf.with_columns(pl.col(feats).cast(pl.Float32)).collect(engine="streaming")
    sub_tr, Xtr, ytr = W.arrays(G, sp["train"], feats)
    sub_va, Xva, yva = W.arrays(G, sp["val"], feats)
    te_sub, Xte, yte = W.arrays(G, sp["test"], feats)
    del G
    mono = list(W.mono_for(feats))
    print(f"rows: train {len(ytr):,} (every second of {len(sp['train'])} matches), val {len(yva):,} ({len(sp['val'])}), "
          f"test {len(yte):,} ({len(sp['test'])}); {len(feats)} inputs ({time.time() - t0:.0f}s)", flush=True)

    out = {"inputs": len(feats), "stride": 1, "train_matches": len(sp["train"]), "val_matches": len(sp["val"]), "test_matches": len(sp["test"]),
           "train_rows": int(len(ytr)), "val_rows": int(len(yva)), "test_rows": int(len(yte)), "models": {}, "device": "gpu"}
    preds = {}

    def record(name, p, fit_s, note, pva):
        p = np.asarray(p, dtype=np.float64)
        preds[name] = p
        r = W.evaluate(te_sub, yte, p, ref=preds["xgb"] if "xgb" in preds and name != "xgb" else None)
        row = {"logloss": r["logloss"], "logloss_ci95": r["logloss_ci95"], "logloss_round_weighted": r["logloss_round_weighted"],
               "logloss_round_weighted_ci95": r["logloss_round_weighted_ci95"], "auc": r["auc"], "ece10": r["ece10"], "fit_seconds": round(fit_s),
               "note": note, "validation_logloss": W.evaluate(sub_va, yva, np.asarray(pva, dtype=np.float64))["logloss"]}
        if name != "xgb" and "delta_vs_ref" in r:
            row["delta_vs_xgb"] = r["delta_vs_ref"]
            row["delta_vs_xgb_ci95"] = r["delta_vs_ref_ci95"]
        out["models"][name] = row
        print(f"{name}: log loss {r['logloss']:.5f} {[round(x, 5) for x in r['logloss_ci95']]}  round-weighted {r['logloss_round_weighted']:.5f}  "
              f"AUC {r['auc']:.4f}  ({fit_s:.0f}s) {note}", flush=True)

    # reference first: the XGBoost fit saved in models/
    import xgboost as xgb
    t = time.time()
    meta = json.load(open(os.path.join(W.ROOT, "models", f"{a.xgb_model}.meta.json"), encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(os.path.join(W.ROOT, "models", f"{a.xgb_model}.json"))
    best = int(meta["best_iteration"])

    def pred_xgb(X):
        return booster.predict(xgb.DMatrix(X), iteration_range=(0, best + 1))

    record("xgb", pred_xgb(Xte), time.time() - t, f"{a.xgb_model}, loaded from models/: best_iteration {best}, fitted on {meta['device']}",
           pva=pred_xgb(Xva))

    def predict_chunked(pred, X, step=1_000_000):
        return np.concatenate([np.asarray(pred(X[i:i + step])) for i in range(0, len(X), step)])

    if "catboost" in names:
        from catboost import CatBoostClassifier
        t = time.time()
        kw = dict(iterations=3000, learning_rate=0.08, depth=6, loss_function="Logloss", eval_metric="Logloss", random_seed=0,
                  od_type="Iter", od_wait=100, use_best_model=True, verbose=0, task_type="GPU", devices="0")
        note = "GPU, depth 6, lr 0.08, early stop on validation"
        try:
            m = CatBoostClassifier(monotone_constraints=mono, **kw)
            m.fit(Xtr, ytr, eval_set=(Xva, yva))
            note += ", monotone constraints on"
        except Exception as exc:  # the GPU may refuse monotone constraints; refit without them and say so in the row
            print(f"catboost on GPU refused the monotone constraints ({str(exc)[:160]}); refitting without them", flush=True)
            m = CatBoostClassifier(**kw)
            m.fit(Xtr, ytr, eval_set=(Xva, yva))
            note += ", NO monotone constraints (the GPU refused them)"
        record("catboost_gpu", m.predict_proba(Xte)[:, 1], time.time() - t, f"{note}, best_iteration {m.get_best_iteration()}",
               pva=m.predict_proba(Xva)[:, 1])
        del m

    if "mlp" in names:
        import torch
        t = time.time()
        dev = torch.device("cuda")
        sel = np.random.default_rng(0).choice(len(ytr), size=int(a.sub_mlp * len(ytr)), replace=False)
        mu = Xtr[sel].mean(axis=0)
        sd = Xtr[sel].std(axis=0)
        sd[sd == 0] = 1.0
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(len(feats), 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(),
                                  torch.nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = torch.nn.BCEWithLogitsLoss()
        Xs = torch.tensor((Xtr[sel] - mu) / sd, dtype=torch.float32, device=dev)
        ys = torch.tensor(ytr[sel], dtype=torch.float32, device=dev)
        bs = 4096
        for _ in range(10):
            perm = torch.randperm(len(ys), device=dev)
            for i in range(0, len(ys), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(net(Xs[idx]).squeeze(1), ys[idx])
                loss.backward()
                opt.step()
        net.eval()

        def pred_mlp(X):
            with torch.no_grad():
                Z = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
                return torch.sigmoid(net(Z).squeeze(1)).cpu().numpy()

        record("mlp_torch_gpu", predict_chunked(pred_mlp, Xte), time.time() - t,
               f"PyTorch on GPU, 128-64 ReLU, Adam 1e-3, batch 4096, 10 epochs, fit on {a.sub_mlp:.0%} of training rows ({len(sel):,})",
               pva=predict_chunked(pred_mlp, Xva))
        del net, Xs, ys
        torch.cuda.empty_cache()

    if "cuml_logit" in names:
        import cupy as cp
        from cuml.linear_model import LogisticRegression as CuLR
        t = time.time()
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0

        def norm(X):
            return cp.asarray(((X - mu) / sd).astype(np.float32))

        m = CuLR(C=1.0, max_iter=300)
        m.fit(norm(Xtr), cp.asarray(ytr.astype(np.float32)))

        def pred_logit(X):
            return to_np(m.predict_proba(norm(X)))[:, 1]

        record("cuml_logit", predict_chunked(pred_logit, Xte), time.time() - t, "cuML LogisticRegression on GPU, standardised, C 1.0, max_iter 300",
               pva=predict_chunked(pred_logit, Xva))
        del m

    if "cuml_rf" in names:
        from cuml.ensemble import RandomForestClassifier as CuRF
        t = time.time()
        sel = np.random.default_rng(0).choice(len(ytr), size=int(a.sub_rf * len(ytr)), replace=False)
        m = CuRF(n_estimators=100, min_samples_leaf=50, random_state=0)
        m.fit(Xtr[sel], ytr[sel].astype(np.int32))

        def pred_rf(X):
            return to_np(m.predict_proba(X))[:, 1]

        record("cuml_rf", predict_chunked(pred_rf, Xte), time.time() - t,
               f"cuML RandomForestClassifier on GPU, 100 trees, min leaf 50, fit on {a.sub_rf:.0%} of training rows ({len(sel):,}); quantile-binned splits, max depth 16 (cuML default)",
               pva=predict_chunked(pred_rf, Xva))
        del m

    ll = sorted(out["models"], key=lambda k: out["models"][k]["logloss"])
    rw = sorted(out["models"], key=lambda k: out["models"][k]["logloss_round_weighted"])
    for i, k in enumerate(ll):
        out["models"][k]["rank_per_second"] = i + 1
        out["models"][k]["rank_round_weighted"] = rw.index(k) + 1
    out["ranking_per_second"] = ll
    out["ranking_round_weighted"] = rw
    W.dump_json(out, os.path.join(d, a.out))
    print(f"ranking, per second: {ll}")
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
