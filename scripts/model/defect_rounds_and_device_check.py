"""NOT RUNNABLE AS WRITTEN (2026-10-25): the 2026 snapshot cache contains none of the roster-defect rounds (251 used rounds, 0.055% of
2026 used rounds, excluded upstream by rounds.use), so the inclusion arm has no rows to add. Its output was discarded (zero defect rows).
The device check is device_parity_check.py. Kept for the record; inclusion would need snapshot features rebuilt for those rounds.

Two checks on the 57-input XGBoost (de_mirage, frozen split, every 10th second, three seeds).

1. Roster-defect rounds. The pinned rule drops every round whose roster lacks a player (wp_common.drop_roster_defect_rounds): about 0.4%
   of 2026 rounds, whose unlisted player's side is inferred and disagrees with the event log in about 40% of cases. Two arms:
     EXCL  train without the defect rounds (the pinned rule)
     INCL  train with them
   Both arms are early-stopped on the same non-defect validation rows and scored on the same non-defect test rows (paired). The
   defect test rows are scored separately, to show the model's log loss where the side inference is wrong.
   Pre-registered: the inclusion effect is material if |mean(INCL - EXCL)| >= 0.0005 and every seed's 95% CI excludes zero (either sign).

2. Device parity. The fits run on CUDA and the prediction path has logged "mismatched devices". Check: the same booster's predictions on
   CUDA and on CPU agree to 1e-6 (max absolute difference) on the test rows.

Writes data/diagnostics/defect_rounds_and_device_check.json.
Run in the cs2gpu environment (polars, xgboost):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/defect_rounds_and_device_check.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "defect_rounds_and_device_check.json")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
STRIDE = 10


def keyed(df):
    return df.with_columns([pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)])


def load(feats):
    need = KEY + ["label_ct_win", "map_name"] + feats
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=sorted(set(need)))
    C = C.filter((pl.col("map_name") == "de_mirage") & pl.col("label_ct_win").is_not_null() & (pl.col("second") % STRIDE == 0))
    R = pl.read_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet"), columns=["match_key", "round", "missing_side"])
    bad = (R.filter(pl.col("missing_side").is_not_null()).select(["match_key", "round"])
             .with_columns(pl.col("round").cast(pl.Int64)).unique().with_columns(pl.lit(True).alias("defect")))
    C = keyed(C).join(bad, on=["match_key", "round"], how="left").with_columns(pl.col("defect").fill_null(False))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    return C.join(A, on="match_key", how="inner").unique(KEY, keep="first", maintain_order=True).sort(KEY)


def fit(tr, va, cols, seed):
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(cols), seed=seed, device="cuda")
    return m, best


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    G = load(feats)
    tr_all = G.filter(pl.col("split") == "train")
    tr_clean = tr_all.filter(~pl.col("defect"))
    va_clean = G.filter((pl.col("split") == "val") & (~pl.col("defect")))
    te = G.filter(pl.col("split") == "test")
    te_nd = te.filter(~pl.col("defect"))
    te_d = te.filter(pl.col("defect"))
    print(f"train rows: all {tr_all.height:,}, clean {tr_clean.height:,}; test rows {te.height:,} "
          f"(defect {te_d.height:,}, matches {int(te_d['match_key'].n_unique())}); {time.time() - t0:.0f} s", flush=True)

    defect_effect = {}
    y_nd = te_nd["label_ct_win"].to_numpy().astype(np.int64)
    match_nd = te_nd["match_key"].to_numpy()
    y_d = te_d["label_ct_win"].to_numpy().astype(np.int64)
    per_seed = {}
    for seed in SEEDS:
        m_ex, b_ex = fit(tr_clean, va_clean, feats, seed)
        m_in, b_in = fit(tr_all, va_clean, feats, seed)
        p_ex = W.predict(m_ex, b_ex, te.select(feats).to_numpy().astype(np.float32))
        p_in = W.predict(m_in, b_in, te.select(feats).to_numpy().astype(np.float32))
        is_nd = (~te["defect"]).to_numpy()
        p_ex_nd, p_in_nd = p_ex[is_nd], p_in[is_nd]
        p_ex_d, p_in_d = p_ex[~is_nd], p_in[~is_nd]
        per_seed[seed] = {
            "INCL_minus_EXCL_nondefect": paired(match_nd, y_nd, p_in_nd, p_ex_nd),
            "logloss_nondefect": {"EXCL": float(W.ll_vec(p_ex_nd, y_nd).mean()), "INCL": float(W.ll_vec(p_in_nd, y_nd).mean())},
            "logloss_defect": {"EXCL": float(W.ll_vec(p_ex_d, y_d).mean()), "INCL": float(W.ll_vec(p_in_d, y_d).mean())},
            "best_iteration": {"EXCL": int(b_ex), "INCL": int(b_in)},
        }
        print(f"seed {seed}: INCL-EXCL (non-defect) {per_seed[seed]['INCL_minus_EXCL_nondefect']['diff']:+.6f}  "
              f"defect rows logloss EXCL {per_seed[seed]['logloss_defect']['EXCL']:.4f} INCL {per_seed[seed]['logloss_defect']['INCL']:.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        if seed == SEEDS[0]:
            Xte = te.select(feats).to_numpy().astype(np.float32)
            parity_model = (m_ex, b_ex)

    d = [per_seed[s]["INCL_minus_EXCL_nondefect"]["diff"] for s in SEEDS]
    ci = [per_seed[s]["INCL_minus_EXCL_nondefect"]["ci95"] for s in SEEDS]
    material = bool(abs(float(np.mean(d))) >= 0.0005 and all((c[0] > 0) or (c[1] < 0) for c in ci))
    defect_effect = {"mean_INCL_minus_EXCL": float(np.mean(d)), "per_seed": d, "per_seed_ci95": ci, "material": material,
                     "defect_test_rows": int(te_d.height), "nondefect_test_rows": int(te_nd.height)}

    # Device parity: the seed-0 EXCL booster, predicted on CUDA (the path the fits use) and on a CPU copy of the same booster.
    m, best = parity_model
    rounds = best + 1
    p_cuda = m.predict_proba(Xte, iteration_range=(0, rounds))[:, 1]
    bst_cpu = xgb.Booster()
    bst_cpu.load_model(bytearray(m.get_booster().save_raw("ubj")))
    bst_cpu.set_param({"device": "cpu"})
    p_cpu = bst_cpu.predict(xgb.DMatrix(Xte), iteration_range=(0, rounds))
    diff = float(np.max(np.abs(p_cuda - p_cpu)))
    parity = {"rows": int(len(Xte)), "max_abs_diff": diff, "agrees_1e6": bool(diff <= 1e-6)}
    print(f"device parity: max abs diff {diff:.2e} over {len(Xte):,} rows (agrees to 1e-6: {parity['agrees_1e6']})", flush=True)

    res = {"seeds": SEEDS, "defect_round_effect": defect_effect, "per_seed": per_seed, "device_parity": parity,
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"defect_round_effect": {k: defect_effect[k] for k in ["mean_INCL_minus_EXCL", "material"]},
                      "device_parity": parity}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
