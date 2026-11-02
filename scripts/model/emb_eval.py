"""Does the learned round-state embedding add information to the 57-input XGBoost? (paired, de_mirage, frozen split, three seeds)

The embedding is z (8 numbers per second) from train_round_state_encoder.py (horizon 3, trained on the train split only, positions
only, no winner and no player identity). Rows are the original de_mirage cache (2026, through 13 October), stride 10.

Arms, all on the same rows:
  A    57 baseline inputs
  E    A + z (8 embedding numbers)
  R    A + 8 Gaussian noise columns (control: the same number of extra columns, carrying no information)
  SP   A + straight-line and walking distances to each site (8 features, from build_nav_path_features.py)
  SPE  SP + z

Pre-registered thresholds (written before any result was read):
  z adds information           if mean(E - A) <= -0.0005 and every seed's 95% upper bound < 0
  z beats noise                if mean(E - R) <= -0.0005 and every seed's 95% upper bound < 0
  z adds beyond the distances  if mean(SPE - SP) <= -0.0005 and every seed's 95% upper bound < 0
Match-cluster bootstrap (spatial_feature_eval.paired, 1000 resamples).

Writes data/diagnostics/emb_eval.json.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "emb_eval.json")
KEY = ["match_key", "round", "second"]
STRIDE = 10
SEEDS = [0, 1, 2]
ZCOLS = [f"z{i}" for i in range(8)]
NCOLS = [f"n{i}" for i in range(8)]
PATH = ["ct_path_siteA", "ct_path_siteB", "t_path_siteA", "t_path_siteB"]
STRAIGHT = ["st_ct_siteA", "st_ct_siteB", "st_t_siteA", "st_t_siteB"]
MISSING = 1.0e4


def base_rows(feats):
    need = KEY + ["label_ct_win", "map_name"] + feats
    G = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=need)
    G = W.drop_roster_defect_rounds(G)
    G = G.filter((pl.col("map_name") == "de_mirage") & pl.col("label_ct_win").is_not_null() & (pl.col("second") % STRIDE == 0))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    return G.join(A, on="match_key", how="inner")


def embedding_frame():
    z = np.load(os.path.join(ROOT, "data", "round_state", "de_mirage_h3.npz"))
    keys = z["match_keys"]
    D = pl.DataFrame({"match_key": keys[z["match_idx"].astype(np.int64)], "round": z["round"].astype(np.int64),
                      "second": z["second"].astype(np.int64)})
    Z = pl.DataFrame({c: z["z"][:, i].astype(np.float64) for i, c in enumerate(ZCOLS)})
    return pl.concat([D, Z], how="horizontal").unique(KEY, keep="first")


def fit(G, cols, seed):
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter(pl.col("split") == "test")
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(cols), seed=seed, device="cuda")
    return te, W.predict(m, best, te.select(cols).to_numpy().astype(np.float32)), best


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    G = base_rows(feats)
    nav = pl.read_parquet(os.path.join(DIAG, "nav_path_de_mirage_stride10.parquet"))
    G = G.join(nav, on=KEY, how="left")
    G = G.with_columns([pl.col(c).fill_null(MISSING).fill_nan(MISSING).clip(upper_bound=MISSING) for c in PATH + STRAIGHT])
    G = G.join(embedding_frame(), on=KEY, how="left")
    n_missing = G.filter(pl.col("z0").is_null()).height
    print(f"rows (stride {STRIDE}): {G.height:,}; rows without an embedding: {n_missing}", flush=True)
    G = G.filter(pl.col("z0").is_not_null())
    rng = np.random.default_rng(12345)
    noise = rng.standard_normal((G.height, 8))
    G = G.with_columns([pl.Series(NCOLS[i], noise[:, i]) for i in range(8)]).sort(KEY)
    arms = {"A": feats, "E": feats + ZCOLS, "R": feats + NCOLS, "SP": feats + STRAIGHT + PATH,
            "SPE": feats + STRAIGHT + PATH + ZCOLS}
    per_seed = {}
    te_ref = None
    for seed in SEEDS:
        preds = {}
        for name, cols in arms.items():
            te_k, p, _ = fit(G, cols, seed)
            preds[name] = p
            te_ref = te_k if te_ref is None else te_ref
        match = te_ref["match_key"].to_numpy()
        y = te_ref["label_ct_win"].to_numpy().astype(np.int64)
        per_seed[seed] = {"E_minus_A": paired(match, y, preds["E"], preds["A"]),
                          "E_minus_R": paired(match, y, preds["E"], preds["R"]),
                          "SPE_minus_SP": paired(match, y, preds["SPE"], preds["SP"]),
                          "R_minus_A": paired(match, y, preds["R"], preds["A"])}
        print(f"seed {seed}: E-A {per_seed[seed]['E_minus_A']['diff']:+.6f}  E-R {per_seed[seed]['E_minus_R']['diff']:+.6f}  "
              f"SPE-SP {per_seed[seed]['SPE_minus_SP']['diff']:+.6f}  R-A {per_seed[seed]['R_minus_A']['diff']:+.6f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    def rule(name):
        d = [per_seed[s][name]["diff"] for s in SEEDS]
        ci = [per_seed[s][name]["ci95"] for s in SEEDS]
        return {"mean_over_seeds": float(np.mean(d)), "per_seed": d, "per_seed_ci95": ci,
                "passes": bool(np.mean(d) <= -0.0005 and all(c[1] < 0 for c in ci))}

    res = {"rows_test": int(te_ref.height), "test_matches": int(np.unique(te_ref["match_key"].to_numpy()).size),
           "rows_without_embedding_dropped": int(n_missing), "seeds": SEEDS, "per_seed": per_seed,
           "verdicts": {"z_adds_information": rule("E_minus_A"), "z_beats_noise": rule("E_minus_R"),
                        "z_adds_beyond_distances": rule("SPE_minus_SP"), "noise_adds_information": rule("R_minus_A")},
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["verdicts"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
