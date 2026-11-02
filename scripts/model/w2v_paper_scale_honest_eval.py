"""Honest paper-scale word2vec: the embedding is fitted on training matches only, and the classifier is trained and tested on match splits.

For each of the 20 samples of 200 de_mirage matches (embedding from w2v_paper_scale_honest_embed.py; 140 training, 60 held-out matches):
  rows     one random second per round; rows whose exact token is out of the training vocabulary are dropped (the paper's rule)
  train    rows of the 140 training matches, with a random 20% of them held out for early stopping (frozen recipe only)
  test     rows of the 60 held-out matches (no row from a training match is used for testing)
  arms     B4 (four base features), E4 (B4 + 8 embedding dims), B57 (frozen 57 features), E57 (B57 + embedding)
  recipes  paper (XGBoost depth 8, default settings, no early stopping) and frozen (W.fit_xgb: depth 4, eta 0.05, mcw 20, subsample 0.8,
           early stopping)

Pre-registered (written before any result was read):
  honest embedding helps the 4-feature baseline (frozen recipe): mean(E4 - B4) >= +0.01 and the 95% samples-bootstrap interval lies above 0
  honest embedding helps beyond 57 (frozen recipe): mean(E57 - B57) >= +0.005 and the interval lies above 0
  paper recipe: mean(E4 - B4) with the same two conditions, reported for comparison with the leaky paper-scale result

Writes data/diagnostics/w2v_paper_scale_honest_eval.json.
Run in the cs2gpu environment (polars, xgboost, CUDA):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/w2v_paper_scale_honest_eval.py
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
SAMPLES = os.path.join(DIAG, "w2v_paper_scale_honest")
OUT = os.path.join(DIAG, "w2v_paper_scale_honest_eval.json")
KEY = ["match_key", "round", "second"]
N_SAMPLES = 20
N_MATCHES = 200
N_TRAIN = 140
BASE4 = ["alive_ct", "alive_t", "st_equip_ct", "st_equip_t"]
EMB = [f"e{i}" for i in range(8)]


def split_sample(k):
    """Same as w2v_paper_scale_honest_embed.split_sample (copied: that module imports gensim, which this environment lacks)."""
    matches = sorted(set(pl.read_parquet(os.path.join(DIAG, "w2v_tokens_de_mirage.parquet"), columns=["match_key"])["match_key"].to_list()))
    rng = np.random.default_rng(1000 + k)
    sample = sorted(rng.choice(matches, size=N_MATCHES, replace=False).tolist())
    perm = np.random.default_rng(3000 + k).permutation(len(sample))
    train = sorted(sample[i] for i in perm[:N_TRAIN])
    held = sorted(sample[i] for i in perm[N_TRAIN:])
    return sample, train, held


def keyed(df):
    return df.with_columns([pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)])


def metrics(p, y):
    pred = (p >= 0.5).astype(np.int64)
    return {"accuracy": float(np.mean(pred == y))}


def paper_fit(tr, te, cols, seed):
    m = XGBClassifier(max_depth=8, eval_metric="logloss", objective="binary:logistic", random_state=seed)
    m.fit(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64))
    return m.predict_proba(te.select(cols).to_numpy().astype(np.float32))[:, 1]


def frozen_fit(tr, va, te, cols, seed):
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(cols), seed=seed, device="cuda")
    return W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))


def boot(v, rng):
    return np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(1000)])


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"),
                        columns=sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + feats)))
    C = W.drop_roster_defect_rounds(C)
    C = keyed(C).filter((pl.col("map_name") == "de_mirage") & pl.col("label_ct_win").is_not_null())
    C = C.unique(KEY, keep="first", maintain_order=True)
    per = []
    for k in range(N_SAMPLES):
        sample, train_m, held_m = split_sample(k)
        E = keyed(pl.read_parquet(os.path.join(SAMPLES, f"sample{k}.parquet")))
        E = E.select(KEY + ["known"] + EMB).unique(KEY, keep="first", maintain_order=True)
        rows = C.filter(pl.col("match_key").is_in(sample))
        rng = np.random.default_rng(2000 + k)
        rows = rows.with_columns(pl.Series("_r", rng.random(rows.height))).sort(["match_key", "round", "_r"])
        rows = rows.unique(["match_key", "round"], keep="first", maintain_order=True).drop("_r")
        d = rows.join(E, on=KEY, how="inner").filter(pl.col("known") == 1).sort(KEY)
        tr_all = d.filter(pl.col("match_key").is_in(train_m))
        te = d.filter(pl.col("match_key").is_in(held_m))
        vperm = np.random.default_rng(4000 + k).permutation(tr_all.height)
        cut = int(round(0.8 * tr_all.height))
        tr_all = tr_all[vperm]
        tr, va = tr_all.slice(0, cut), tr_all.slice(cut)
        y = te["label_ct_win"].to_numpy().astype(np.int64)
        out = {"sample": k, "rows_train": tr.height, "rows_val": va.height, "rows_test": te.height}
        for name, cols in {"B4": BASE4, "E4": BASE4 + EMB, "B57": feats, "E57": feats + EMB}.items():
            out[f"paper_{name}_acc"] = metrics(paper_fit(tr_all, te, cols, k), y)["accuracy"]
            out[f"frozen_{name}_acc"] = metrics(frozen_fit(tr, va, te, cols, k), y)["accuracy"]
        per.append(out)
        print(f"sample {k}: train {tr.height:,}, held-out test {te.height:,}; frozen B4 {out['frozen_B4_acc']:.4f} "
              f"E4 {out['frozen_E4_acc']:.4f} B57 {out['frozen_B57_acc']:.4f} E57 {out['frozen_E57_acc']:.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    rng = np.random.default_rng(0)
    summary = {}
    for recipe in ["frozen", "paper"]:
        for lhs, rhs in [("E4", "B4"), ("E57", "B57")]:
            v = np.array([r[f"{recipe}_{lhs}_acc"] - r[f"{recipe}_{rhs}_acc"] for r in per])
            bs = boot(v, rng)
            summary[f"{recipe}_{lhs}_minus_{rhs}"] = {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    verdicts = {
        "frozen_embedding_helps_4_features": bool(summary["frozen_E4_minus_B4"]["mean"] >= 0.01 and summary["frozen_E4_minus_B4"]["ci95"][0] > 0),
        "frozen_embedding_helps_beyond_57": bool(summary["frozen_E57_minus_B57"]["mean"] >= 0.005 and summary["frozen_E57_minus_B57"]["ci95"][0] > 0),
        "paper_embedding_helps_4_features": bool(summary["paper_E4_minus_B4"]["mean"] >= 0.01 and summary["paper_E4_minus_B4"]["ci95"][0] > 0),
    }
    res = {"samples": N_SAMPLES, "per_sample": per, "summary": summary, "pre_registered_verdicts": verdicts,
           "mean_test_rows": float(np.mean([r["rows_test"] for r in per])), "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"summary": summary, "verdicts": verdicts}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
