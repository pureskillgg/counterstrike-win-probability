"""Plain MLP on the 57 frozen inputs, all maps, three seeds, compared with the all-maps XGBoost arm A (same inputs, same rows).

Config (fixed in the pre-registration, Meridian, before any run):
  inputs     the 57 frozen features, standardised with train-row mean and std (fitted on train rows only)
  net        57 -> 256 -> 256 -> 1, ReLU, dropout 0.2 after each hidden layer, BCE-with-logits
  optimiser  AdamW, lr 1e-3, weight decay 1e-4, batch 4096
  training   up to 15 epochs; early stopping with patience 2 on validation log loss; best checkpoint kept
  rows       all training rows (all seven maps); roster-defect rounds excluded exactly as in wp_common.drop_roster_defect_rounds
  no monotone constraints (XGBoost has them; the difference is stated, not tuned away)

DECISION (validation seconds only): the MLP beats arm A if, for EACH seed, mean(MLP minus A) on the validation seconds is <= -0.0005
with a 95% match-cluster CI upper bound < 0, and the three-seed mean is <= -0.0005. The test seconds (de_mirage, 865,974 seconds) are
scored and saved for the record; they play no part in the decision.

Comparator: arm A predictions from scripts/model/sitecheck_xgb.py (data/diagnostics/preds/allmaps_xgb_A_seed<s>.parquet). The script
stops if they are missing. Rows are aligned on match_key, round, second; the number of aligned rows is reported.

Writes data/diagnostics/preds/allmaps_mlp_seed<s>.parquet and data/diagnostics/allmaps_mlp_vs_xgbA.json.
Run in the torch environment (CUDA; no xgboost needed):
  G:/pureskill/csds-cs2/.pixi/envs/torch/python.exe -I -X utf8 -u scripts/model/mlp_allmaps.py
"""
import copy
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

ROOT = os.environ.get("CS2_ROOT", "G:/pureskill/csds-cs2")
DATA = os.path.join(ROOT, "data")
DIAG = os.path.join(DATA, "diagnostics")
PREDS = os.path.join(DIAG, "preds")
OUT = os.path.join(DIAG, "allmaps_mlp_vs_xgbA.json")
MAPS_TEST = "de_mirage"
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
BATCH = 4096
MAX_EPOCHS = 15
PATIENCE = 2
CHUNK = 2_000_000


def logloss_t(y, p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return float(-(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean())


def bootstrap_paired(match, y, p_a, p_b, seed=0):
    """Mean of (log loss of a minus log loss of b) over seconds, match-cluster bootstrap with 1,000 resamples."""
    pa, pb = np.clip(p_a, 1e-6, 1 - 1e-6), np.clip(p_b, 1e-6, 1 - 1e-6)
    diff = -(y * np.log(pa) + (1 - y) * np.log(1 - pa)) + (y * np.log(pb) + (1 - y) * np.log(1 - pb))
    uniq, inv = np.unique(match, return_inverse=True)
    sums, counts = np.zeros(uniq.size), np.zeros(uniq.size)
    np.add.at(sums, inv, diff)
    np.add.at(counts, inv, 1)
    rng = np.random.default_rng(seed)
    boot = np.empty(1000)
    for i in range(1000):
        pick = rng.integers(0, uniq.size, uniq.size)
        boot[i] = sums[pick].sum() / counts[pick].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "rows": int(diff.size), "matches": int(uniq.size)}


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    cols = sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + feats))
    C = pl.read_parquet(os.path.join(DATA, "combined", "cache_snapshot_2026_ext.parquet"), columns=cols)
    C = C.with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    R = pl.read_parquet(os.path.join(DATA, "combined", "rounds.parquet"), columns=["match_key", "round", "missing_side"])
    bad = R.filter(pl.col("missing_side").is_not_null()).select("match_key", "round").with_columns(pl.col("round").cast(pl.Int64))
    C = C.join(bad, on=["match_key", "round"], how="anti")
    C = C.filter(pl.col("label_ct_win").is_not_null()).unique(KEY, keep="first", maintain_order=True).sort(KEY)
    man = json.load(open(os.path.join(DATA, "combined", "split_manifest.json"), encoding="utf-8"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    G = C.join(A, on="match_key", how="inner")
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter((pl.col("split") == "test") & (pl.col("map_name") == MAPS_TEST))
    print(f"rows: train {tr.height:,}, val {va.height:,}, test {te.height:,} ({time.time() - t0:.0f} s)", flush=True)

    stats = tr.select([pl.col(c).cast(pl.Float64).mean().alias(c + "__m") for c in feats] +
                      [pl.col(c).cast(pl.Float64).std().alias(c + "__s") for c in feats]).row(0, named=True)
    mu = np.array([stats[c + "__m"] for c in feats], dtype=np.float32)
    sd = np.maximum(np.array([stats[c + "__s"] for c in feats], dtype=np.float32), 1e-6)
    dev = torch.device("cuda")

    def to_device(frame):
        n = frame.height
        X = torch.empty((n, len(feats)), dtype=torch.float32, device=dev)
        y = torch.empty(n, dtype=torch.float32, device=dev)
        for s in range(0, n, CHUNK):
            part = frame.slice(s, CHUNK)
            x = (part.select(feats).to_numpy().astype(np.float32) - mu) / sd
            X[s:s + part.height] = torch.from_numpy(x).to(dev)
            y[s:s + part.height] = torch.from_numpy(part["label_ct_win"].to_numpy().astype(np.float32)).to(dev)
        return X, y

    X_tr, y_tr = to_device(tr)
    X_va, y_va = to_device(va)
    X_te, y_te = to_device(te)
    print(f"on device ({time.time() - t0:.0f} s)", flush=True)

    @torch.no_grad()
    def predict(model, X):
        model.eval()
        out = torch.empty(X.shape[0], device=dev)
        for s in range(0, X.shape[0], 65536):
            out[s:s + 65536] = torch.sigmoid(model(X[s:s + 65536]).squeeze(1))
        model.train()
        return out

    os.makedirs(PREDS, exist_ok=True)
    per_seed = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        model = nn.Sequential(nn.Linear(57, 256), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(256, 1)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()
        best = (float("inf"), None, 0)
        bad_epochs = 0
        history = []
        for epoch in range(1, MAX_EPOCHS + 1):
            perm = torch.randperm(X_tr.shape[0], device=dev, generator=gen)
            running = torch.zeros((), device=dev)
            for s in range(0, X_tr.shape[0], BATCH):
                idx = perm[s:s + BATCH]
                loss = bce(model(X_tr[idx]).squeeze(1), y_tr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.detach() * idx.shape[0]
            train_ll = float(running) / X_tr.shape[0]
            v = logloss_t(y_va, predict(model, X_va))
            history.append({"epoch": epoch, "train_loss": train_ll, "val_log_loss": v})
            print(f"seed {seed} epoch {epoch}: train {train_ll:.5f}, validation {v:.5f} ({time.time() - t0:.0f} s)", flush=True)
            if v < best[0]:
                best = (v, copy.deepcopy(model.state_dict()), epoch)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    break
        model.load_state_dict(best[1])
        p_va = predict(model, X_va).cpu().numpy().astype(np.float64)
        p_te = predict(model, X_te).cpu().numpy().astype(np.float64)
        pl.concat([
            pl.DataFrame({"match_key": va["match_key"].to_numpy(), "round": va["round"].to_numpy(), "second": va["second"].to_numpy(),
                          "map_name": va["map_name"].to_numpy(), "split": "val", "label": va["label_ct_win"].to_numpy().astype(np.int64), "p": p_va}),
            pl.DataFrame({"match_key": te["match_key"].to_numpy(), "round": te["round"].to_numpy(), "second": te["second"].to_numpy(),
                          "map_name": te["map_name"].to_numpy(), "split": "test", "label": te["label_ct_win"].to_numpy().astype(np.int64), "p": p_te}),
        ]).write_parquet(os.path.join(PREDS, f"allmaps_mlp_seed{seed}.parquet"))
        per_seed.append({"seed": seed, "selected_epoch": best[2], "history": history, "val_log_loss": best[0],
                         "xgb_A_file": os.path.join(PREDS, f"allmaps_xgb_A_seed{seed}.parquet")})

    results = []
    for seed, rec in zip(SEEDS, per_seed):
        xp = rec["xgb_A_file"]
        if not os.path.exists(xp):
            raise SystemExit(f"missing comparator predictions {xp}; run sitecheck_xgb.py first")
        mp = pl.read_parquet(os.path.join(PREDS, f"allmaps_mlp_seed{seed}.parquet"))
        xp_df = pl.read_parquet(xp)
        out = {"seed": seed, "selected_epoch": rec["selected_epoch"], "history": rec["history"]}
        for split in ["val", "test"]:
            m = mp.filter(pl.col("split") == split)
            x = xp_df.filter(pl.col("split") == split)
            j = m.join(x.select(KEY + ["p"]).rename({"p": "p_xgb"}), on=KEY, how="inner")
            d = bootstrap_paired(j["match_key"].to_numpy(), j["label"].to_numpy().astype(np.float64),
                                 j["p"].to_numpy(), j["p_xgb"].to_numpy())
            out[f"mlp_minus_xgbA_{split}"] = d
            out[f"aligned_rows_{split}"] = j.height
            out[f"mlp_log_loss_{split}"] = float(np.mean(-(j["label"].to_numpy() * np.log(np.clip(j["p"].to_numpy(), 1e-6, 1 - 1e-6))
                                                          + (1 - j["label"].to_numpy()) * np.log(np.clip(1 - j["p"].to_numpy(), 1e-6, 1 - 1e-6)))))
            out[f"xgbA_log_loss_{split}"] = float(np.mean(-(j["label"].to_numpy() * np.log(np.clip(j["p_xgb"].to_numpy(), 1e-6, 1 - 1e-6))
                                                           + (1 - j["label"].to_numpy()) * np.log(np.clip(1 - j["p_xgb"].to_numpy(), 1e-6, 1 - 1e-6)))))
        out["passes_validation_this_seed"] = bool(out["mlp_minus_xgbA_val"]["diff"] <= -0.0005 and out["mlp_minus_xgbA_val"]["ci95"][1] < 0)
        results.append(out)
        print(json.dumps({k: out[k] for k in ["seed", "passes_validation_this_seed"]}), flush=True)

    mean_val = float(np.mean([r["mlp_minus_xgbA_val"]["diff"] for r in results]))
    decision = bool(all(r["passes_validation_this_seed"] for r in results) and mean_val <= -0.0005)
    res = {"seeds": results, "mean_mlp_minus_xgbA_validation": mean_val,
           "mlp_beats_xgbA_on_validation": decision, "decision_set": "validation seconds only",
           "seconds": round(time.time() - t0)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ["mean_mlp_minus_xgbA_validation", "mlp_beats_xgbA_on_validation"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
