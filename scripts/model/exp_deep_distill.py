"""Deep teacher and teacher-student distillation against XGBoost, on the same held-out matches (GPU: PyTorch + XGBoost).

  python -X utf8 scripts/model/exp_deep_distill.py [--stride 2] [--folds 3] [--epochs 14] [--width 512] [--max-rows 0]

Question: does a DEEP model (an MLP over the same 59 inputs as the final XGBoost) beat boosted trees, and does a student XGBoost trained on the deep
teacher's probabilities beat the XGBoost trained on the hard labels? Everything is scored on the frozen-split TEST matches with paired match-cluster
intervals (wp_common.evaluate with ref=...), so every difference is a paired difference on identical seconds.

Protocol (leak-free):
  * rows: the 2026 snapshot cache, TRAINING matches only for fitting, VALIDATION matches for early stopping, TEST matches only for the final scores.
  * teacher: an MLP (SiLU, LayerNorm, dropout) trained with log loss. Out-of-fold probabilities for the training rows come from K group folds BY MATCH
    (fold = hash of the match key): a row's teacher probability never comes from a net that saw its match. Teacher predictions for validation and test
    rows are the mean of the K fold nets.
  * student: XGBoost (native API, same settings as the final model: depth 4, eta 0.05, min_child_weight 20, subsample 0.8, monotone constraints on the
    alive counts) trained on soft labels  alpha * y + (1 - alpha) * p_teacher_oof, early-stopped on the validation matches' TRUE labels. alpha = 1 is the
    ordinary hard-label model (the reference of every difference).
  * extra rows: the teacher alone, and the plain average of teacher and hard-label XGBoost.
Writes data/<store>/exp_deep_distill.json.
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import hashlib
import json
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]


def fold_of(keys, k):
    return np.array([int(hashlib.sha256(str(m).encode()).hexdigest()[:8], 16) % k for m in keys])


class MLP(nn.Module):
    def __init__(self, d_in, width, depth, drop):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(drop)]
            d = width
            width = max(128, width // 2)
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict(model, X, dev, bs=262144):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(X[i:i + bs].to(dev))).float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def fit_mlp(Xtr, ytr, Xva, yva, dev, epochs, width, depth, drop, lr, bs, seed, wd=1e-4, evals_per_epoch=1, noise=0.0):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], width, depth, drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    per_epoch = (len(Xtr) + bs - 1) // bs
    steps = epochs * per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    Xg, yg = Xtr.to(dev), ytr.to(dev)
    eval_every = max(1, per_epoch // max(1, evals_per_epoch))
    best, best_state, hist, step = 9.0, None, [], 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xg), device=dev)
        for i in range(0, len(Xg), bs):
            idx = perm[i:i + bs]
            xb = Xg[idx]
            if noise:
                xb = xb + noise * torch.randn_like(xb)  # input noise: a cheap regulariser against memorising near-identical seconds
            loss = F.binary_cross_entropy_with_logits(model(xb), yg[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % eval_every == 0 or step == steps:
                p = predict(model, Xva, dev)
                model.train()
                ll = float(W.ll_vec(p, yva.numpy()).mean())
                hist.append(round(ll, 5))
                if ll < best:
                    best, best_state = ll, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--alphas", default="0.0,0.5")
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--evals-per-epoch", type=int, default=1, help="validate this many times per epoch (the first run overfit after one epoch)")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise (in standard deviations) added to the standardised inputs while training")
    ap.add_argument("--tag", default="", help="suffix of the results file, so several configurations can be kept")
    ap.add_argument("--max-rows", type=int, default=0, help="debug: cap the training rows")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", a.store)
    G = pl.read_parquet(os.path.join(d, "cache_snapshot_2026.parquet"))
    sp = W.make_split("random", G, seed=0)
    ST = [c for c in G.columns if c.startswith("st_")]
    feats = W.F6 + [f"map_is_{m}" for m in MAPS] + ST
    sub_tr, Xtr, ytr = W.arrays(G, sp["train"], feats, stride=a.stride)
    sub_va, Xva, yva = W.arrays(G, sp["val"], feats, stride=a.stride)
    sub_te, Xte, yte = W.arrays(G, sp["test"], feats)
    if a.max_rows:
        Xtr, ytr, sub_tr = Xtr[:a.max_rows], ytr[:a.max_rows], sub_tr.head(a.max_rows)
    print(f"device {dev}; features {len(feats)}; train {len(ytr):,} rows ({len(sp['train'])} matches), val {len(yva):,}, test {len(yte):,} ({len(sp['test'])} matches); {time.time() - t0:.0f}s", flush=True)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda X: torch.from_numpy(((X - mu) / sd).astype(np.float32))
    Ttr, Tva, Tte = norm(Xtr), norm(Xva), norm(Xte)
    ytr_t, yva_t = torch.from_numpy(ytr.astype(np.float32)), torch.from_numpy(yva.astype(np.float32))

    # --- teacher: K group folds by match -> out-of-fold probabilities for the training rows; validation / test = mean of the fold nets
    fold = fold_of(sub_tr["match_key"].to_numpy(), a.folds)
    p_oof, p_va, p_te, hists = np.zeros(len(ytr)), np.zeros(len(yva)), np.zeros(len(yte)), []
    for k in range(a.folds):
        m = fold != k
        model, best, hist = fit_mlp(Ttr[torch.from_numpy(m)], ytr_t[torch.from_numpy(m)], Tva, yva_t, dev, a.epochs, a.width, a.depth, a.dropout, a.lr, a.batch, seed=k,
                                    wd=a.wd, evals_per_epoch=a.evals_per_epoch, noise=a.noise)
        p_oof[~m] = predict(model, Ttr[torch.from_numpy(~m)], dev)
        p_va += predict(model, Tva, dev) / a.folds
        p_te += predict(model, Tte, dev) / a.folds
        hists.append(hist)
        print(f"teacher fold {k}: best val LL {best:.4f} per epoch {hist} ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    ll = lambda p, y: float(W.ll_vec(p, y).mean())
    print(f"teacher out-of-fold LL on train rows {ll(p_oof, ytr):.4f}; val (mean of nets) {ll(p_va, yva):.4f}", flush=True)

    # --- students: XGBoost on soft labels (alpha = 1 is the hard-label reference)
    mono = tuple(W.mono_for(feats))
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device=dev, max_depth=4, eta=0.05, min_child_weight=20,
                  subsample=0.8, monotone_constraints=mono, seed=0)

    def fit_student(alpha):
        soft = alpha * ytr + (1.0 - alpha) * p_oof
        dtr = xgb.QuantileDMatrix(Xtr, label=soft)
        dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr)
        bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
        return bst, bst.predict(xgb.DMatrix(Xte), iteration_range=(0, bst.best_iteration + 1))

    res = {"protocol": vars(a), "device": dev, "train_rows": int(len(ytr)), "test_rows": int(len(yte)), "test_matches": len(sp["test"]), "teacher_val_curves": hists}
    bst1, p_hard = fit_student(1.0)
    print(f"hard-label XGBoost: best iteration {bst1.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    keep = ("logloss", "logloss_ci95", "auc", "brier", "ece10", "ece10_ci95", "smooth_ece")
    r = W.evaluate(sub_te, yte, p_hard)
    res["xgboost_hard_labels"] = {k: r[k] for k in keep}
    rows = {"deep_teacher (mean of fold nets)": p_te, "average of teacher and hard-label XGBoost": 0.5 * (p_te + p_hard)}
    for al in [float(x) for x in a.alphas.split(",") if x]:
        b, p = fit_student(al)
        rows[f"student XGBoost, soft labels alpha={al:g}"] = p
        print(f"student alpha={al:g}: best iteration {b.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    for name, p in rows.items():
        r = W.evaluate(sub_te, yte, np.clip(p, 1e-6, 1 - 1e-6), ref=p_hard)
        res[name] = {**{k: r[k] for k in keep}, "delta_vs_xgboost_hard": r["delta_vs_ref"], "delta_ci95": r["delta_vs_ref_ci95"]}
        print(f"{name}: LL {r['logloss']:.4f}  vs hard-label XGBoost {r['delta_vs_ref']:+.4f} {[round(x, 4) for x in r['delta_vs_ref_ci95']]}  AUC {r['auc']:.4f} ECE {r['ece10']:.4f}", flush=True)
    res["seconds"] = round(time.time() - t0)
    W.dump_json(res, os.path.join(d, f"exp_deep_distill{('_' + a.tag) if a.tag else ''}.json"))


if __name__ == "__main__":
    main()
