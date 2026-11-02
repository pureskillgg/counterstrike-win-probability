"""Does the honest word2vec embedding add information to the 57-input XGBoost, and to the distance features? (paired, de_mirage, three seeds)

Same rows, base and distances as emb_eval.py (every 10th second of the frozen split, original de_mirage cache).
The embedding is the 8-dimensional word2vec replication fitted on TRAIN rounds only (word2vec_replication.py --mode honest).
Known limit: most seconds get the zero vector (unknown token), so the embedding is mostly zeros on these rows.

Arms: A (57), W (A + 8 word2vec dims), SP (A + straight-line and walking distances), SPW (SP + word2vec dims).

Pre-registered thresholds (written before any result was read):
  word2vec adds information   if mean(W - A) <= -0.0005 and every seed's 95% upper bound < 0
  word2vec adds beyond distances if mean(SPW - SP) <= -0.0005 and every seed's 95% upper bound < 0
Writes data/diagnostics/w2v_eval2.json.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402
import emb_eval as E  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

DIAG = E.DIAG
OUT = os.path.join(DIAG, "w2v_eval2.json")
WCOLS = [f"w2v{i}" for i in range(8)]
SEEDS = [0, 1, 2]


def w2v_frame():
    w = pl.read_parquet(os.path.join(DIAG, "w2v_de_mirage_honest.parquet"))
    return w.select(E.KEY + WCOLS).unique(E.KEY, keep="first")


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(E.ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    G = E.base_rows(feats)
    nav = pl.read_parquet(os.path.join(DIAG, "nav_path_de_mirage_stride10.parquet"))
    G = G.join(nav, on=E.KEY, how="left")
    G = G.with_columns([pl.col(c).fill_null(E.MISSING).fill_nan(E.MISSING).clip(upper_bound=E.MISSING) for c in E.PATH + E.STRAIGHT])
    G = G.join(w2v_frame(), on=E.KEY, how="left").with_columns([pl.col(c).fill_null(0.0) for c in WCOLS])
    zero_share = float(G.filter((pl.col("w2v0") == 0) & (pl.col("w2v1") == 0)).height / G.height)
    print(f"rows {G.height:,}; share of rows with the zero (unknown-token) embedding: {zero_share:.3f}", flush=True)
    G = G.sort(E.KEY)
    arms = {"A": feats, "W": feats + WCOLS, "SP": feats + E.STRAIGHT + E.PATH, "SPW": feats + E.STRAIGHT + E.PATH + WCOLS}
    per_seed = {}
    te_ref = None
    for seed in SEEDS:
        preds = {}
        for name, cols in arms.items():
            te_k, p, _ = E.fit(G, cols, seed)
            preds[name] = p
            te_ref = te_k if te_ref is None else te_ref
        match = te_ref["match_key"].to_numpy()
        y = te_ref["label_ct_win"].to_numpy().astype(np.int64)
        per_seed[seed] = {"W_minus_A": paired(match, y, preds["W"], preds["A"]),
                          "SPW_minus_SP": paired(match, y, preds["SPW"], preds["SP"])}
        print(f"seed {seed}: W-A {per_seed[seed]['W_minus_A']['diff']:+.6f}  SPW-SP {per_seed[seed]['SPW_minus_SP']['diff']:+.6f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    def rule(name):
        d = [per_seed[s][name]["diff"] for s in SEEDS]
        ci = [per_seed[s][name]["ci95"] for s in SEEDS]
        return {"mean_over_seeds": float(np.mean(d)), "per_seed": d, "per_seed_ci95": ci,
                "passes": bool(np.mean(d) <= -0.0005 and all(c[1] < 0 for c in ci))}

    res = {"zero_embedding_share": zero_share, "rows_test": int(te_ref.height), "per_seed": per_seed,
           "verdicts": {"w2v_adds_information": rule("W_minus_A"), "w2v_adds_beyond_distances": rule("SPW_minus_SP")},
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["verdicts"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
