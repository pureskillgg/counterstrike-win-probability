"""Calibration of the shipped 83-input model on the full validation set, and the 1-versus-5 states.

Why: the explanation of the largest miss rests on one 1-versus-5 state. Seconds within a round share one outcome, so this
script checks that state per round as well as per second:
  - the shipped booster's validation log loss (should match the refit's 0.450210 to about 1e-6)
  - a decile calibration table over all validation seconds: mean predicted P(CT wins) against the observed CT-win rate, with
    Wilson 95% intervals. These intervals treat seconds as independent, so they are too narrow.
  - the 1-versus-5 states (CT 1 alive, T 5 alive; bomb planted, no bomb, or either) per round: expected CT-win rounds is the sum
    of each round's mean predicted P(CT wins) over its seconds in the state; observed is the number of rounds CT won.

Reads the matrix written by sprint_prepare.py and the booster in models/. Writes the tables under docs/explainability/tables/.
Run with the pixi environment:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe scripts\\explain\\xgb83_calibration.py
"""
import json
import os
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgb83_explain import DOCS, MDIR, WORK  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BOOSTER = os.path.join(REPO, "models", "winprob_xgb83_final.json")
META = os.path.join(REPO, "models", "winprob_xgb83_final.meta.json")
TABLES = os.path.join(DOCS, "tables")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(centre - half), float(centre + half))


def main():
    meta = json.load(open(os.path.join(MDIR, "meta.json")))
    feats = meta["feats_all"]
    a2 = meta["n_fit"] + meta["n_es"]
    best = int(json.load(open(META))["delivered_file"]["best_iteration"])
    X = np.load(os.path.join(MDIR, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(MDIR, "y.npy"), mmap_mode="r")
    Xv = np.asarray(X[a2:], dtype=np.float32)
    yv = np.asarray(y[a2:], dtype=np.float64)
    n = len(yv)
    assert n == meta["n_val"], "validation row count differs from meta.json"

    p_path = os.path.join(WORK, "p_val_shipped.npy")
    if os.path.exists(p_path) and np.load(p_path).shape[0] == n:
        p = np.load(p_path).astype(np.float64)
    else:
        bst = xgb.Booster()
        bst.load_model(BOOSTER)
        p = bst.predict(xgb.DMatrix(Xv), iteration_range=(0, best + 1)).astype(np.float64)
        os.makedirs(WORK, exist_ok=True)
        np.save(p_path, p.astype(np.float32))
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    ll = float(-(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)).mean())
    print(f"shipped booster, {n:,} validation seconds: log loss {ll:.6f}", flush=True)

    # decile calibration over all seconds
    order = np.argsort(p)
    rows = []
    for k, idx in enumerate(np.array_split(order, 10)):
        kk = int(yv[idx].sum())
        lo, hi = wilson(kk, len(idx))
        rows.append({"decile": k + 1, "n": int(len(idx)), "mean_predicted_p_ct_win": float(p[idx].mean()),
                     "observed_ct_win_rate": kk / len(idx), "observed_ci_low": lo, "observed_ci_high": hi,
                     "min_p": float(p[idx].min()), "max_p": float(p[idx].max())})
    pl.DataFrame(rows).write_csv(os.path.join(TABLES, "calibration_deciles_shipped.csv"))

    # 1-versus-5 states, per second and per round
    ac, at, bp = (np.asarray(X[a2:, feats.index(f)], dtype=np.float64) for f in ["alive_ct", "alive_t", "bomb_planted"])
    keys = pl.read_parquet(os.path.join(MDIR, "val_keys.parquet"))
    assert keys.height == n and (keys["label_ct_win"].to_numpy() == yv).all(), "val_keys do not line up with the validation rows"
    dfv = keys.with_columns(pl.Series("p", p), pl.Series("ac", ac), pl.Series("at", at), pl.Series("bp", bp))
    base = (pl.col("ac") == 1) & (pl.col("at") == 5)
    states = {"1v5_any_bomb": base, "1v5_bomb_planted": base & (pl.col("bp") == 1), "1v5_no_bomb": base & (pl.col("bp") == 0)}
    srows = []
    for name, cond in states.items():
        s = dfv.filter(cond)
        r = s.group_by(["match_key", "round"]).agg(pl.col("p").mean().alias("mean_p"),
                                                    pl.col("label_ct_win").max().alias("ct_win"), pl.len().alias("secs"))
        kk = int(s["label_ct_win"].sum())
        lo, hi = wilson(kk, s.height)
        exp_rounds, obs_rounds = float(r["mean_p"].sum()), int(r["ct_win"].sum())
        srows.append({"state": name, "rounds": r.height, "matches": int(r["match_key"].n_unique()),
                      "seconds": s.height, "expected_ct_win_rounds": exp_rounds, "observed_ct_win_rounds": obs_rounds,
                      "ct_win_seconds": kk, "mean_predicted_p_ct_win_per_second": float(s["p"].mean()),
                      "observed_ct_win_rate_per_second": kk / s.height, "per_second_ci_low": lo, "per_second_ci_high": hi})
        print(srows[-1], flush=True)
    pl.DataFrame(srows).write_csv(os.path.join(TABLES, "calibration_1v5_shipped.csv"))
    json.dump({"shipped_validation_logloss": ll, "n_validation": n, "best_iteration": best,
               "decile_table": "calibration_deciles_shipped.csv", "state_table": "calibration_1v5_shipped.csv"},
              open(os.path.join(WORK, "calibration_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
