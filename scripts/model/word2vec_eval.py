"""Evaluate the word2vec replication against the 59-feature baseline on de_mirage (XGBoost, GPU, paired).

Arms, all trained and tested on the SAME rows (the seconds that have a token), in the same order:
  A1  baseline inputs without map one-hots (de_mirage train only)            <- the reference
  B   A1 + 8 word2vec dimensions fitted on train rounds only (honest)
  C   A1 + 8 random dimensions (negative control)
  L   A1 + 8 word2vec dimensions fitted on every round (leaky; the published setup)

Pre-registered thresholds (set before any result was read):
  B adds information  if mean(B - A1) <= -0.0005 and the 95% upper bound < 0.
  C adds nothing      if |mean(C - A1)| <= 0.0005.
  Leakage explains part of the published gain if mean(L - B) <= -0.0005 and its upper bound < 0.

  ...cs2gpu python.exe -I -X utf8 scripts\\model\\word2vec_eval.py
Writes data/diagnostics/w2v_eval.json.
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
OUT = os.path.join(DIAG, "w2v_eval.json")
N_BOOT = 1000
STRIDE = 2
KEY = ["match_key", "round", "second"]


def base_rows():
    # the reference is the 57-input model (winprob_v6_drop_heavy): no score lead, no heavy-weapon counts. Its map one-hots are
    # constant on de_mirage, so they are kept as they are in the reference.
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    need = KEY + ["label_ct_win", "map_name"] + feats
    G = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=need)
    G = W.drop_roster_defect_rounds(G)
    G = G.filter((pl.col("map_name") == "de_mirage") & pl.col("label_ct_win").is_not_null())
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    G = G.join(A, on="match_key", how="inner")
    return G, feats


def emb(mode):
    return pl.read_parquet(os.path.join(DIAG, f"w2v_de_mirage_{mode}.parquet"))


def paired(match, y, pa, pb, seed=0):
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
    return {"mean_a": float(la.mean()), "mean_b": float(lb.mean()), "diff": float(la.mean() - lb.mean()),
            "ci95": [float(lo), float(hi)], "rows": int(len(y)), "matches": int(M)}


def fit_arm(G, cols, name):
    tr = G.filter((pl.col("split") == "train") & (pl.col("second") % STRIDE == 0))
    va = G.filter(pl.col("split") == "val")
    te = G.filter(pl.col("split") == "test")
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(cols), seed=0, device="cuda")
    pte = W.predict(m, best, te.select(cols).to_numpy().astype(np.float32))
    print(f"  {name}: best iteration {best}, test rows {te.height:,}", flush=True)
    return te, pte, best


def main():
    t0 = time.time()
    G, feats = base_rows()
    w2v = [f"w2v{j}" for j in range(8)]
    tokens = emb("honest").select(KEY)  # every arm uses exactly the seconds that have a token
    Gc = G.join(tokens, on=KEY, how="inner").sort(KEY)
    print(f"common rows: {Gc.height:,} seconds, {Gc['match_key'].n_unique()} matches", flush=True)
    te_a1, p_a1, best_a1 = fit_arm(Gc, feats, "A1 baseline")
    preds = {"A1": p_a1}
    best = {"A1": best_a1}
    for mode in ["honest", "random", "leaky"]:
        Gm = Gc.join(emb(mode), on=KEY, how="inner").sort(KEY)
        assert Gm.height == Gc.height and (Gm.select(KEY).to_numpy() == Gc.select(KEY).to_numpy()).all(), "row alignment"
        te_m, p_m, best_m = fit_arm(Gm, feats + w2v, mode)
        preds[mode] = p_m
        best[mode] = best_m
    te = te_a1  # the same test rows, in the same order, for every arm
    match = te["match_key"].to_numpy()
    y = te["label_ct_win"].to_numpy().astype(np.int64)
    res = {
        "B_minus_A1": paired(match, y, preds["honest"], preds["A1"]),
        "C_minus_A1": paired(match, y, preds["random"], preds["A1"]),
        "L_minus_A1": paired(match, y, preds["leaky"], preds["A1"]),
        "L_minus_B": paired(match, y, preds["leaky"], preds["honest"]),
        "best_iteration": best,
        "thresholds": {"B_adds_information": "mean(B-A1) <= -0.0005 and CI upper < 0",
                       "C_adds_nothing": "|mean(C-A1)| <= 0.0005",
                       "leakage_explains_gain": "mean(L-B) <= -0.0005 and CI upper < 0"},
    }
    b = res["B_minus_A1"]
    c = res["C_minus_A1"]
    lb = res["L_minus_B"]
    res["verdicts"] = {
        "B_adds_information": bool(b["diff"] <= -0.0005 and b["ci95"][1] < 0),
        "C_adds_nothing": bool(abs(c["diff"]) <= 0.0005),
        "leakage_explains_gain": bool(lb["diff"] <= -0.0005 and lb["ci95"][1] < 0),
    }
    res["seconds"] = round(time.time() - t0)
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
