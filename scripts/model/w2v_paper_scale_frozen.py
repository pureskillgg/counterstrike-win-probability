"""Paper scale again, with our frozen training recipe and a 57-feature baseline: does the word2vec embedding add anything?

Same 20 samples of 200 de_mirage matches as w2v_paper_scale_eval.py (embeddings from w2v_paper_scale_embed.py, fitted on each sample).
Same rows: one random second per round, unknown tokens dropped, random 80/20 row split. The 20% validation split for early stopping is
carved from the training rows (random, seed k).

Arms (all on the same kept rows):
  B4    the four base features (num_t_alive, num_ct_alive, t_equip_value, ct_equip_value)
  E4    B4 + the 8 embedding dimensions
  B57   the 57 frozen features of winprob_v6_drop_heavy
  E57   B57 + the 8 embedding dimensions
Training recipe (the frozen one, W.fit_xgb): depth 4, eta 0.05, min_child_weight 20, subsample 0.8, up to 3000 trees with early stopping
on the validation rows, monotone constraints from W.mono_for. The paper's recipe (depth 8, no early stopping) is in
w2v_paper_scale_eval.py and is not repeated here.

Pre-registered (written before any result was read):
  embedding adds at paper scale on the 4-feature baseline: mean(E4 - B4) >= +0.01 and the 95% samples-bootstrap interval lies above zero
  embedding adds beyond the 57-feature baseline: mean(E57 - B57) >= +0.005 and the interval lies above zero
  the 57-feature baseline is stronger than the 4-feature one: mean(B57 - B4) > 0 (a sanity check on the recipe)
Full scale (57 features, same recipe, de_mirage 2026 frozen split) is already in w2v_eval2.json: honest embedding, mean +0.00012.

Writes data/diagnostics/w2v_paper_scale_frozen.json.
Run in the cs2gpu environment (polars, xgboost, CUDA):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/w2v_paper_scale_frozen.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
SAMPLES = os.path.join(DIAG, "w2v_paper_scale")
OUT = os.path.join(DIAG, "w2v_paper_scale_frozen.json")
KEY = ["match_key", "round", "second"]
N_SAMPLES = 20
BASE4 = ["alive_ct", "alive_t", "st_equip_ct", "st_equip_t"]
EMB = [f"e{i}" for i in range(8)]


def keyed(df):
    return df.with_columns([pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)])


def fit_predict(tr, va, te, cols, seed):
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(cols), seed=seed, device="cuda")
    return W.predict(m, best, te.select(cols).to_numpy().astype(np.float32)), best


def metrics(p, y):
    pred = (p >= 0.5).astype(np.int64)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    return {"accuracy": float(np.mean(pred == y)), "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def boot(diffs, rng):
    return np.array([diffs[rng.integers(0, diffs.size, diffs.size)].mean() for _ in range(1000)])


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    need = sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + feats))
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=need)
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
        d = rows.join(E, on=KEY, how="inner").filter(pl.col("known") == 1).sort(KEY)
        perm = rng.permutation(d.height)
        cut = int(round(0.8 * d.height))
        dp = d[perm]
        trainval, te = dp.slice(0, cut), dp.slice(cut)
        vcut = int(round(0.8 * trainval.height))
        tr, va = trainval.slice(0, vcut), trainval.slice(vcut)
        y = te["label_ct_win"].to_numpy().astype(np.int64)
        arms = {"B4": BASE4, "E4": BASE4 + EMB, "B57": feats, "E57": feats + EMB}
        preds, best, mets = {}, {}, {}
        for name, cols in arms.items():
            preds[name], best[name] = fit_predict(tr, va, te, cols, k)
            mets[name] = metrics(preds[name], y)
        per.append({"sample": k, "rows_kept": d.height, "n_train": tr.height, "n_val": va.height, "n_test": te.height,
                    **{f"{n}_acc": mets[n]["accuracy"] for n in arms}, **{f"{n}_f1": mets[n]["f1"] for n in arms},
                    "best_iteration": best,
                    "E4_minus_B4": mets["E4"]["accuracy"] - mets["B4"]["accuracy"],
                    "E57_minus_B57": mets["E57"]["accuracy"] - mets["B57"]["accuracy"],
                    "B57_minus_B4": mets["B57"]["accuracy"] - mets["B4"]["accuracy"]})
        print(f"sample {k}: test {te.height:,}; acc B4 {mets['B4']['accuracy']:.4f} E4 {mets['E4']['accuracy']:.4f} "
              f"B57 {mets['B57']['accuracy']:.4f} E57 {mets['E57']['accuracy']:.4f} ({time.time() - t0:.0f} s)", flush=True)

    rng = np.random.default_rng(0)
    summary = {}
    for name in ["E4_minus_B4", "E57_minus_B57", "B57_minus_B4"]:
        v = np.array([r[name] for r in per])
        bs = boot(v, rng)
        summary[name] = {"mean": float(v.mean()), "ci95_samples_bootstrap": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    rule = {
        "embedding_adds_on_4_features": bool(summary["E4_minus_B4"]["mean"] >= 0.01 and summary["E4_minus_B4"]["ci95_samples_bootstrap"][0] > 0),
        "embedding_adds_beyond_57": bool(summary["E57_minus_B57"]["mean"] >= 0.005 and summary["E57_minus_B57"]["ci95_samples_bootstrap"][0] > 0),
        "57_beats_4": bool(summary["B57_minus_B4"]["mean"] > 0),
    }
    res = {"samples": N_SAMPLES, "recipe": "frozen (depth 4, eta 0.05, mcw 20, subsample 0.8, early stopping on 20% of train)",
           "per_sample": per, "summary": summary, "pre_registered_verdicts": rule,
           "mean_accuracy": {n: float(np.mean([r[f"{n}_acc"] for r in per])) for n in ["B4", "E4", "B57", "E57"]},
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"mean_accuracy": res["mean_accuracy"], "summary": summary, "verdicts": rule}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
