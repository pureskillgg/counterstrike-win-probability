"""Does the word2vec gain appear at the paper's scale? The Chahrour (2023) protocol on 20 random samples of 200 de_mirage matches.

Per sample k (embedding from w2v_paper_scale_embed.py, fitted on that sample only):
  rows     one random second per round of the sample's matches (the paper's rule), with the round winner as the label
  kept     rows whose exact token is in the sample's vocabulary (the paper drops the rest)
  arms     B  = the four base features (num_t_alive, num_ct_alive, t_equip_value, ct_equip_value)
           E  = B + the 8 embedding dimensions
  split    random 80/20 by row
  model    XGBoost max_depth 8, other settings default (the paper's settings), accuracy and F1 at 0.5 on the test rows
Paper (Chahrour 2023, verified from the page images): baseline accuracy 0.6477, with embedding 0.67486 (+0.027), F1 0.70 -> 0.65.

Pre-registered (written before any result was read):
  the gain reproduces at paper scale if the mean over the 20 samples of (E - B) accuracy is >= +0.01 and its 95% interval (samples
  bootstrap, 1,000 resamples, seed 0) lies above zero.
  the baseline level reproduces if the mean B accuracy is within 0.02 of 0.6477.
Reported with the full-scale result for comparison (w2v_published_eval, de_mirage, 5,498 matches).

Writes data/diagnostics/w2v_paper_scale_eval.json.
Run in the cs2gpu environment (polars, xgboost):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/w2v_paper_scale_eval.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
SAMPLES = os.path.join(DIAG, "w2v_paper_scale")
OUT = os.path.join(DIAG, "w2v_paper_scale_eval.json")
KEY = ["match_key", "round", "second"]
N_SAMPLES = 20
BASE = ["alive_ct", "alive_t", "st_equip_ct", "st_equip_t"]
EMB = [f"e{i}" for i in range(8)]
PAPER = {"baseline_acc": 0.6477, "embedding_acc": 0.67486}


def keyed(df):
    return df.with_columns([pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)])


def fit_predict(tr, te, cols, seed):
    m = XGBClassifier(max_depth=8, eval_metric="logloss", objective="binary:logistic", random_state=seed)
    m.fit(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64))
    return m.predict_proba(te.select(cols).to_numpy().astype(np.float32))[:, 1]


def metrics(p, y):
    pred = (p >= 0.5).astype(np.int64)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    return {"accuracy": float(np.mean(pred == y)), "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def main():
    t0 = time.time()
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"),
                        columns=sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + BASE)))
    C = W.drop_roster_defect_rounds(C)
    C = keyed(C).filter((pl.col("map_name") == "de_mirage") & pl.col("label_ct_win").is_not_null())
    C = C.unique(KEY, keep="first", maintain_order=True)
    print(f"de_mirage cache rows {C.height:,} ({time.time() - t0:.0f} s)", flush=True)

    per = []
    for k in range(N_SAMPLES):
        E = keyed(pl.read_parquet(os.path.join(SAMPLES, f"sample{k}.parquet")))
        matches = E["match_key"].unique().to_list()
        E = E.select(KEY + ["known"] + EMB).unique(KEY, keep="first", maintain_order=True)
        rows = C.filter(pl.col("match_key").is_in(matches))
        rng = np.random.default_rng(2000 + k)
        rows = rows.with_columns(pl.Series("_r", rng.random(rows.height))).sort(["match_key", "round", "_r"])
        rows = rows.unique(["match_key", "round"], keep="first", maintain_order=True).drop("_r")
        d = rows.join(E, on=KEY, how="inner").filter(pl.col("known") == 1)
        d = d.sort(KEY)
        perm = rng.permutation(d.height)
        cut = int(round(0.8 * d.height))
        dp = d[perm]
        tr, te = dp.slice(0, cut), dp.slice(cut)
        y = te["label_ct_win"].to_numpy().astype(np.int64)
        pB = fit_predict(tr, te, BASE, k)
        pE = fit_predict(tr, te, BASE + EMB, k)
        mB, mE = metrics(pB, y), metrics(pE, y)
        per.append({"sample": k, "matches": len(matches), "rows_kept": d.height, "n_train": tr.height, "n_test": te.height,
                    "B_acc": mB["accuracy"], "E_acc": mE["accuracy"], "B_f1": mB["f1"], "E_f1": mE["f1"],
                    "E_minus_B_acc": mE["accuracy"] - mB["accuracy"], "E_minus_B_f1": mE["f1"] - mB["f1"]})
        print(f"sample {k}: kept {d.height:,}, test {te.height:,}; acc B {mB['accuracy']:.4f} E {mE['accuracy']:.4f} "
              f"(E-B {mE['accuracy'] - mB['accuracy']:+.4f}); F1 B {mB['f1']:.4f} E {mE['f1']:.4f} ({time.time() - t0:.0f} s)",
              flush=True)

    diffs = np.array([r["E_minus_B_acc"] for r in per])
    f1d = np.array([r["E_minus_B_f1"] for r in per])
    rng = np.random.default_rng(0)
    boot = np.array([diffs[rng.integers(0, diffs.size, diffs.size)].mean() for _ in range(1000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    B_mean = float(np.mean([r["B_acc"] for r in per]))
    res = {"samples": N_SAMPLES, "matches_per_sample": 200, "per_sample": per,
           "mean_B_acc": B_mean, "mean_E_acc": float(np.mean([r["E_acc"] for r in per])),
           "mean_E_minus_B_acc": float(diffs.mean()), "ci95_samples_bootstrap": [float(lo), float(hi)],
           "mean_E_minus_B_f1": float(f1d.mean()),
           "gain_reproduces": bool(diffs.mean() >= 0.01 and lo > 0),
           "baseline_level_reproduces": bool(abs(B_mean - PAPER["baseline_acc"]) <= 0.02),
           "paper": PAPER, "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("mean_B_acc", "mean_E_acc", "mean_E_minus_B_acc", "ci95_samples_bootstrap",
                                           "mean_E_minus_B_f1", "gain_reproduces", "baseline_level_reproduces")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
