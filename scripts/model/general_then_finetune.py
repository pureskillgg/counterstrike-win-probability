"""General deep model on the 59 baseline inputs, then per-map fine-tuning for the active-duty maps.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\torch\\python.exe -I -X utf8 -u scripts\\model\\general_then_finetune.py

1. General model: a small MLP fitted on the train split of ALL maps (the baseline's own features, standardised on train),
   early-stopped on the all-map validation split.
2. Fine-tuned models: for each active-duty map, a copy of the general model is trained further on that map's train matches only,
   with a lower learning rate, and early-stopped on that map's validation matches.
3. Scoring: each test second is scored by the general model, by the fine-tuned model of its map, and by the frozen baseline export
   (winprob_v6_drop_heavy) on the same rows. Differences use a 1,000-resample match-cluster bootstrap (seed 0).

Frozen split: data/combined/split_manifest.json. Inputs: the 2026 cache (cache_snapshot_2026.parquet, all maps).
Writes data/diagnostics/general_then_finetune.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import torch
import torch.nn as nn

import wp_common as W

OUT = os.path.join(W.ROOT, "data", "diagnostics", "general_then_finetune.json")
ACTIVE = ["de_ancient", "de_anubis", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]  # confirm against the map_name values
N_BOOT = 1000


def load():
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    need = ["match_key", "round", "tick", "second", "label_ct_win", "map_name"] + feats
    G = pl.read_parquet(os.path.join(W.ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=need)
    G = W.drop_roster_defect_rounds(G)
    man = W.load_manifest(os.path.join(W.ROOT, "data", "combined", "split_manifest.json"))
    A = pl.DataFrame({"match_key": list(man["assign"].keys()), "split": list(man["assign"].values())})
    G = G.join(A, on="match_key", how="inner").filter(pl.col("label_ct_win").is_not_null())
    E = pl.read_parquet(os.path.join(W.ROOT, "data", "combined", "export_winprob_v6_drop_heavy_test_predictions.parquet"))
    E = E.select(["match_key", "round", "tick", pl.col("p").alias("p_base")])
    G = G.join(E, on=["match_key", "round", "tick"], how="left")  # the export covers the test rows only; train and val keep null p_base
    return G, feats


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def fit(model, Xtr, ytr, Xva, yva, lr, epochs, patience, dev, batch=4096):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.as_tensor(Xtr, device=dev)
    yt = torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    Xv = torch.as_tensor(Xva, device=dev)
    best, best_state, bad, ep = np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for s in range(0, len(Xt), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv)).cpu().numpy()
        v = float(W.ll_vec(pv, yva).mean())
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep


def predict(model, X, dev):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, device=dev))).cpu().numpy()


def paired_boot(match, y, pa, pb, seed=0):
    """Mean log loss of pa, of pb, and the paired difference pa - pb with a match-cluster bootstrap interval."""
    la = W.ll_vec(pa, y)
    lb = W.ll_vec(pb, y)
    uniq, inv = np.unique(match, return_inverse=True)
    M = len(uniq)
    cnt = np.bincount(inv, minlength=M).astype(float)
    sa = np.bincount(inv, weights=la, minlength=M)
    sb = np.bincount(inv, weights=lb, minlength=M)
    rng = np.random.default_rng(seed)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, M, M), minlength=M).astype(float)
        d[b] = ((w * sa).sum() - (w * sb).sum()) / (w * cnt).sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"logloss": float(la.mean()), "logloss_baseline": float(lb.mean()), "diff": float(la.mean() - lb.mean()),
            "diff_ci95": [float(lo), float(hi)], "rows": int(len(y)), "matches": int(M)}


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    G, feats = load()
    print(f"rows {G.height:,}; device {dev}; maps {G['map_name'].unique().sort().to_list()}", flush=True)
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter((pl.col("split") == "test") & pl.col("p_base").is_not_null())  # scored only where the baseline has a prediction
    Xtr = tr.select(feats).to_numpy().astype(np.float32)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd < 1e-6] = 1.0

    def X(df):
        return ((df.select(feats).to_numpy().astype(np.float32) - mu) / sd).astype(np.float32)

    Xtr_s, ytr = X(tr), tr["label_ct_win"].to_numpy().astype(np.float32)
    Xva_s, yva = X(va), va["label_ct_win"].to_numpy().astype(np.float32)
    Xte_s, yte = X(te), te["label_ct_win"].to_numpy().astype(np.float32)
    general = Net(len(feats)).to(dev)
    general, gbest, gep = fit(general, Xtr_s, ytr, Xva_s, yva, lr=1e-3, epochs=30, patience=3, dev=dev)
    print(f"general: best val log loss {gbest:.5f} at epoch {gep}  ({time.time() - t0:.0f}s)", flush=True)
    p_gen = predict(general, Xte_s, dev)

    p_ft = np.full(te.height, np.nan)
    mt = te["map_name"].to_numpy()
    ft_info = {}
    for m in ACTIVE:
        trm = tr.filter(pl.col("map_name") == m)
        vam = va.filter(pl.col("map_name") == m)
        if trm.height < 1000 or vam.height < 500:
            ft_info[m] = {"skipped": f"train rows {trm.height}, val rows {vam.height}"}
            continue
        ft = Net(len(feats)).to(dev)
        ft.load_state_dict({k: v.clone() for k, v in general.state_dict().items()})
        ft, fbest, fep = fit(ft, X(trm), trm["label_ct_win"].to_numpy().astype(np.float32),
                             X(vam), vam["label_ct_win"].to_numpy().astype(np.float32),
                             lr=1e-4, epochs=8, patience=2, dev=dev, batch=2048)
        sel = mt == m
        p_ft[sel] = predict(ft, Xte_s[sel], dev)
        ft_info[m] = {"train_rows": trm.height, "train_matches": trm["match_key"].n_unique(), "val_best": fbest, "epochs": fep}
        print(f"fine-tuned {m}: train matches {trm['match_key'].n_unique()}, best val {fbest:.5f} at epoch {fep}", flush=True)

    match = te["match_key"].to_numpy()
    pb = te["p_base"].to_numpy().astype(np.float64)
    results = {"overall_all_maps": {"general_vs_baseline": paired_boot(match, yte, p_gen, pb)}}
    per_map = {}
    for m in sorted(set(mt.tolist())):
        sel = mt == m
        entry = {"active": m in ACTIVE, "general_vs_baseline": paired_boot(match[sel], yte[sel], p_gen[sel], pb[sel])}
        if m in ACTIVE and not np.isnan(p_ft[sel]).any():
            entry["finetuned_vs_baseline"] = paired_boot(match[sel], yte[sel], p_ft[sel], pb[sel])
            entry["finetuned_vs_general"] = paired_boot(match[sel], yte[sel], p_ft[sel], p_gen[sel])
        per_map[m] = entry
    results["per_map"] = per_map
    results["fine_tune"] = ft_info
    results["general_best_val"] = gbest
    results["settings"] = {"features": len(feats), "general": "MLP 59-256-128-1, AdamW 1e-3, early stop 3",
                           "fine_tune": "lr 1e-4, up to 8 epochs, early stop 2, per active map", "bootstrap": N_BOOT, "seed": 0,
                           "test_rows": te.height, "test_matches": te["match_key"].n_unique()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)", flush=True)
    for m, e in per_map.items():
        ft = e.get("finetuned_vs_baseline")
        print(f"{m:12s} general {e['general_vs_baseline']['diff']:+.5f} {e['general_vs_baseline']['diff_ci95']}"
              + (f" | finetuned {ft['diff']:+.5f} {ft['diff_ci95']}" if ft else ""), flush=True)


if __name__ == "__main__":
    main()
