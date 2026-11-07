"""A9 audit: re-run cross-period model A on the 2024-02 kept set, and measure the round-13 defect's effect.

Read-only. Imports scripts/model/sufficiency/cross_period.py (its main() is not run) and reuses
run_2026() and run_2024() unchanged, so the 2024 kept set is the one the reported numbers used.

Steps
  1. Rebuild the 2026 model-A training set and the 2024 kept set (cross_period.run_2026/run_2024).
  2. Fit model A on the 2026 primary train+test (1399 matches), early stop on the 122-match slice,
     device cpu with bake-off default threads (cross_period used device "cuda", which falls back to
     CPU in this environment). Predict every 2024 kept state. Compare with the JSON
     (logloss 0.5044104148798078, best_iteration 265).
  3. Round-13 defect (open, see scripts/swarm/e2_side_check_r13.py): for each 2024 kept round with a
     bomb outcome, bomb_side = 3 (CT) if bomb_defused else 2 (T) for bomb_exploded; a round is
     flagged if bomb_side != winner_team_code. Flags are restricted to the kept rounds.
  4. Log loss on 2024 states: all; excluding flagged rounds; flagged rounds with the labels set to
     the bomb-consistent side. ASSUMPTION: the bomb outcome is the correct side (a bomb that explodes
     is a T win), so a flagged round's winner_team_code is the suspect field.

Usage (project root): python.exe -B -I -X utf8 scripts\\swarm2\\A9_cross_period_r13.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
CP_PATH = ROOT / "scripts" / "model" / "sufficiency" / "cross_period.py"
JSON_LL_B = 0.5044104148798078
JSON_BEST_A = 265


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_a(bk, A):
    Xtr = A["S_trA"].select(bk.FEATS).to_numpy().astype(np.float64)
    ytr = A["S_trA"]["label_ct_win"].to_numpy().astype(np.int64)
    Xva = A["S_vaA"].select(bk.FEATS).to_numpy().astype(np.float64)
    yva = A["S_vaA"]["label_ct_win"].to_numpy().astype(np.int64)
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                          subsample=0.8, tree_method="hist", device="cpu",
                          monotone_constraints=tuple(bk.MONO), early_stopping_rounds=100,
                          eval_metric="logloss", random_state=bk.SEED)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, int(m.best_iteration)


def defect_flags(cp):
    fv = cp.fv
    old = (fv.TOME, fv.SUFFIX)
    fv.TOME, fv.SUFFIX = cp.TOME_2024, cp.SUFFIX_2024
    try:
        RE = fv.read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
        BS = fv.read("bomb_state", ["match_key", "round", "event_type"])
    finally:
        fv.TOME, fv.SUFFIX = old
    RE = RE.unique(["match_key", "round"], keep="first")
    outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    j = (RE.join(outcome, on=["match_key", "round"], how="inner")
         .with_columns(
             pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2))
             .alias("bomb_side"))
         .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
    return j


def main():
    t0 = time.time()
    cp = load_module("cross_period", CP_PATH)
    bk = cp.bk
    A = cp.run_2026()
    B = cp.run_2024()
    print(f"2024 counts from run_2024: {B['counts']}", flush=True)
    print(f"2026 model-A sizes: {A['split_sizes']}", flush=True)

    mA, bi = fit_a(bk, A)
    S24 = B["S"]
    X24 = S24.select(bk.FEATS).to_numpy().astype(np.float64)
    y24 = S24["label_ct_win"].to_numpy().astype(np.int64)
    p24 = mA.predict_proba(X24, iteration_range=(0, bi + 1))[:, 1]
    met = bk.metrics(y24, p24)
    print(f"model A: best_iter {bi} (JSON {JSON_BEST_A}); 2024 all-state LL {met['logloss']:.10f} "
          f"(JSON {JSON_LL_B:.10f}; diff {met['logloss'] - JSON_LL_B:+.2e}); "
          f"Brier {met['brier']:.6f}", flush=True)

    j = defect_flags(cp)
    kept_rounds = S24.select(["match_key", "round"]).unique()
    jk = j.join(kept_rounds, on=["match_key", "round"], how="inner")
    flagged = jk.filter(pl.col("mismatch"))
    print(f"2024 kept rounds: {kept_rounds.height}; with a bomb outcome: {jk.height}; "
          f"flagged (bomb side != winner_team_code): {flagged.height}", flush=True)
    print("flagged by round: " + str(flagged.group_by("round").len().sort("round").rows()[:20]), flush=True)
    print("flagged by day: " + str(flagged.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1)
                                                     .alias("day"))
                                    .group_by("day").len().sort("day").rows()), flush=True)
    r13 = jk.filter(pl.col("round") == 13)
    print(f"round 13 with outcome: {r13.height}; flagged {int(r13['mismatch'].sum())}", flush=True)
    print("flagged by player_count: " + str(flagged.group_by("player_count").len().rows()), flush=True)

    fl = flagged.select(["match_key", "round", "bomb_side"])
    key = S24.select(["match_key", "round"]).with_row_index("i")
    fs = key.join(fl, on=["match_key", "round"], how="inner")
    idx_f = fs["i"].to_numpy()
    mask_f = np.zeros(len(y24), dtype=bool)
    mask_f[idx_f] = True
    bomb_lab = np.zeros(len(y24), dtype=np.int64)
    bomb_lab[idx_f] = (fs["bomb_side"].to_numpy() == 3).astype(np.int64)
    print(f"flagged states: {int(mask_f.sum())} of {len(y24)} ({mask_f.mean():.4%})", flush=True)

    ll_all = bk.metrics(y24, p24)["logloss"]
    ll_excl = bk.metrics(y24[~mask_f], p24[~mask_f])["logloss"]
    y_fix = y24.copy()
    y_fix[mask_f] = bomb_lab[mask_f]
    ll_fix = bk.metrics(y_fix, p24)["logloss"]
    ll_flag_only = bk.metrics(y24[mask_f], p24[mask_f])["logloss"]
    ll_flag_fix = bk.metrics(y_fix[mask_f], p24[mask_f])["logloss"]
    print(f"2024 LL all states           {ll_all:.6f}  (n={len(y24)})", flush=True)
    print(f"2024 LL excluding flagged    {ll_excl:.6f}  (n={int((~mask_f).sum())})", flush=True)
    print(f"2024 LL with flagged labels set to bomb side {ll_fix:.6f}", flush=True)
    print(f"flagged states LL with winner labels {ll_flag_only:.6f}; with bomb labels {ll_flag_fix:.6f}",
          flush=True)
    print(f"share of flagged states with winner label != bomb label: "
          f"{float((y24[mask_f] != bomb_lab[mask_f]).mean()):.4f}", flush=True)
    print(f"2026 in-period reference (model B on primary test) 0.4366; gap all-state {ll_all - 0.43660:+.5f}, "
          f"excluding flagged {ll_excl - 0.43660:+.5f}, bomb-label fix {ll_fix - 0.43660:+.5f}", flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
