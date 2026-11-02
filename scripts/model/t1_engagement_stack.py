"""T1 (Meridian pin fe9246e1): does a per-second death-in-the-next-3-seconds probability, stacked into the distance XGBoost, improve validation log loss?

Target: for each second (row) on the all-maps population, 1 if any player dies at a tick in (t, t + 192], where t is the row's tick
(64 ticks per second, so 192 ticks is 3 seconds). Deaths from data/combined/deaths.parquet, all sides (no side filter; the pin says "any player").

Stacking guard (as pinned): the engagement classifier is XGBoost on the 57 frozen inputs plus the four train-centre distances, fitted on
train matches only. Training rows get out-of-fold predictions from match-grouped 5-fold CV inside the training split. Validation and test rows
get predictions from the classifier fitted on all training rows. A row never sees a model that trained on its own match.

Implementation choices the pin leaves open (fixed here, before any run):
  - classifier recipe: 400 trees, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, hist, CUDA, no early stopping. Early stopping on the
    validation split would let validation labels shape the stacked feature, so it is not used.
  - fold split: match-grouped, 5 folds, assigned with numpy seed 0 (the same folds for every seed).
  - classifier random_state = the seed of the round model (0, 1, 2).
  - round model: W.fit_xgb with the same recipe and early stopping on the validation split as every other arm, on the 57 inputs, the four
    distances and the engagement probability (62 inputs).
  - the comparison arm S is the saved SN arm of sitecheck_xgb.py (57 inputs plus train-centre distances), same seeds.

Rule (pin fe9246e1): PASS only if for EACH seed mean(S plus engagement minus S) on validation is at most -0.0005 with a 95% match-clustered CI
upper bound below 0, and the three-seed mean is at most -0.0005. Test (de_mirage) is descriptive only.

Writes data/diagnostics/t1_engagement_stack.json and data/diagnostics/preds/t1_se_seed<s>_val.parquet. Run in the cs2gpu environment:
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe -I -X utf8 -u scripts/model/t1_engagement_stack.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp_common as W  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
PREDS = os.path.join(DIAG, "preds")
DEATHS = os.path.join(ROOT, "data", "combined", "deaths.parquet")
OUT = os.path.join(DIAG, "t1_engagement_stack.json")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEY = ["match_key", "round", "second"]
ST = [f"st{i}" for i in range(4)]
NST = [f"nst{i}" for i in range(4)]
SEEDS = [0, 1, 2]
N_FOLDS = 5
TICKS_3S = 192                       # 3 seconds at 64 ticks per second
TICK_STRIDE = 2 ** 32                # keeps (round group, tick) keys apart in one int64
ENG = dict(n_estimators=400, learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
           tree_method="hist", device="cuda")


def sorted_join(base, frame, label_col):
    """Align frame (one row per key) to base (sorted by KEY): returns the matching rows in base order, checking labels agree."""
    J = base.join(frame.sort(KEY).rename({label_col: "label_ref"}), on=KEY, how="inner", maintain_order="left")
    assert J.height == base.height, f"{J.height:,} matched rows, expected {base.height:,}"
    assert (J["label_ref"].to_numpy().astype(np.int64) == base["label"].to_numpy().astype(np.int64)).all(), "labels do not align"
    return J


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    S_cols = feats + NST
    cols = S_cols + ["eng"]
    geo_old = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo", f"{m}.parquet")) for m in MAPS])
    geo_new = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo_trainsites", f"{m}.parquet")) for m in MAPS])
    geo_new = geo_new.rename({f"st{i}": f"nst{i}" for i in range(4)}).unique(KEY, keep="first", maintain_order=True)
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"),
                        columns=sorted(set(KEY + ["match_key", "tick", "label_ct_win", "map_name"] + feats)))
    C = W.drop_roster_defect_rounds(C)
    C = C.filter(pl.col("label_ct_win").is_not_null()).with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    # the all-maps population of sitecheck_xgb.py: a second needs a row in both geometry sets, then the split assignment.
    # The population keys are built on the narrow columns first; the full-width cache columns are gathered once, afterwards.
    pop = (C.select(KEY + ["map_name", "label_ct_win"])
             .join(geo_old.select(KEY).unique(KEY), on=KEY, how="semi")
             .join(geo_new.select(KEY + NST), on=KEY, how="inner")
             .join(A, on="match_key", how="inner"))
    pop = pop.unique(KEY, keep="first", maintain_order=True).sort(KEY)
    G = pop.join(C.select(KEY + ["tick"] + feats), on=KEY, how="left", maintain_order="left")
    del C, geo_old, geo_new, pop
    print(f"population rows {G.height:,} ({time.time() - t0:.0f} s)", flush=True)

    # engagement target: any death at a tick in (t, t + 192] in the same round
    deaths = pl.read_parquet(DEATHS, columns=["match_key", "round", "tick"]).with_columns(pl.col("round").cast(pl.Int64))
    gk = deaths.select(["match_key", "round"]).unique().sort(["match_key", "round"]).with_row_index("gid")
    dkeys = (deaths.join(gk, on=["match_key", "round"], how="inner")
                   .select((pl.col("gid").cast(pl.Int64) * TICK_STRIDE + pl.col("tick")).alias("key"))
                   .sort("key")["key"].to_numpy())
    rows_g = G.select(["match_key", "round", "tick"]).join(gk, on=["match_key", "round"], how="left", maintain_order="left")
    gid = rows_g["gid"].fill_null(-1).to_numpy().astype(np.int64)
    tick = rows_g["tick"].to_numpy().astype(np.int64)
    lo = gid * TICK_STRIDE + tick
    hi = lo + TICKS_3S
    cnt = np.searchsorted(dkeys, hi, side="right") - np.searchsorted(dkeys, lo, side="right")
    y_eng = ((cnt > 0) & (gid >= 0)).astype(np.int8)
    del deaths, gk, dkeys, rows_g, gid, tick, lo, hi, cnt
    print(f"engagement target: positive rate {y_eng.mean():.4f} over {y_eng.size:,} rows ({time.time() - t0:.0f} s)", flush=True)

    # match-grouped folds inside the training split (fixed seed 0)
    train_matches = G.filter(pl.col("split") == "train")["match_key"].unique().sort().to_list()
    rng = np.random.default_rng(0)
    fm = pl.DataFrame({"match_key": train_matches, "fold": (rng.permutation(len(train_matches)) % N_FOLDS).astype(np.int64)})
    fold = G.select("match_key").join(fm, on="match_key", how="left", maintain_order="left")["fold"].fill_null(-1).to_numpy()
    is_train = (G["split"] == "train").to_numpy()
    is_val = (G["split"] == "val").to_numpy()
    is_test = ((G["split"] == "test") & (G["map_name"] == "de_mirage")).to_numpy()
    rest = np.flatnonzero(is_val | is_test)
    tr_idx, va_idx, te_idx = np.flatnonzero(is_train), np.flatnonzero(is_val), np.flatnonzero(is_test)
    y_all = G["label_ct_win"].to_numpy().astype(np.int64)

    # feature matrix: 57 inputs and four distances, plus one column for the engagement probability (filled per seed).
    # Filled one column at a time, so no 61-column temporary is ever built.
    X_SE = np.empty((G.height, len(cols)), dtype=np.float32)
    for j, c in enumerate(S_cols):
        X_SE[:, j] = G[c].to_numpy().astype(np.float32)
    base_va = G.filter(pl.col("split") == "val").select(KEY + ["label_ct_win"]).rename({"label_ct_win": "label"})
    base_te = (G.filter((pl.col("split") == "test") & (pl.col("map_name") == "de_mirage"))
                 .select(KEY + ["label_ct_win"]).rename({"label_ct_win": "label"}))
    del G
    print(f"features ready ({time.time() - t0:.0f} s)", flush=True)

    res = {"per_seed": {}, "rule": "per seed mean(S+eng minus S) on validation <= -0.0005 with CI upper < 0; three-seed mean <= -0.0005"}
    per_seed_pass = []
    diffs = []
    for s in SEEDS:
        eng = X_SE[:, len(S_cols)]
        eng[:] = 0.0
        for k in range(N_FOLDS):
            tr_k = np.flatnonzero(is_train & (fold != k))
            te_k = np.flatnonzero(is_train & (fold == k))
            clf = xgb.XGBClassifier(**ENG, random_state=s)
            clf.fit(X_SE[tr_k, :len(S_cols)], y_eng[tr_k])
            eng[te_k] = clf.predict_proba(X_SE[te_k, :len(S_cols)])[:, 1]
            print(f"seed {s} fold {k}: out-of-fold rows {te_k.size:,} ({time.time() - t0:.0f} s)", flush=True)
            del clf
        clf = xgb.XGBClassifier(**ENG, random_state=s)
        clf.fit(X_SE[tr_idx, :len(S_cols)], y_eng[tr_idx])
        eng[rest] = clf.predict_proba(X_SE[rest, :len(S_cols)])[:, 1]
        del clf
        print(f"seed {s}: engagement predictions done ({time.time() - t0:.0f} s)", flush=True)

        m, best = W.fit_xgb(X_SE[tr_idx], y_all[tr_idx], X_SE[va_idx], y_all[va_idx], mono=W.mono_for(cols), seed=s, device="cuda")
        p_va = W.predict(m, best, X_SE[va_idx])
        p_te = W.predict(m, best, X_SE[te_idx])
        print(f"seed {s}: S plus engagement best iteration {best} ({time.time() - t0:.0f} s)", flush=True)

        # comparison arm S: the saved SN predictions of sitecheck_xgb.py, aligned to the same validation and de_mirage test rows
        sn = pl.read_parquet(os.path.join(PREDS, f"allmaps_xgb_SN_seed{s}.parquet"))
        J_va = sorted_join(base_va, sn.filter(pl.col("split") == "val").select(KEY + ["label", "p"]), "label")
        J_te = sorted_join(base_te, sn.filter(pl.col("split") == "test").select(KEY + ["label", "p"]), "label")
        y_va = base_va["label"].to_numpy().astype(np.int64)
        y_te = base_te["label"].to_numpy().astype(np.int64)
        d_va = paired(base_va["match_key"].to_numpy(), y_va, p_va, J_va["p"].to_numpy())
        d_te = paired(base_te["match_key"].to_numpy(), y_te, p_te, J_te["p"].to_numpy())
        pl.DataFrame({"match_key": base_va["match_key"].to_numpy(), "round": base_va["round"].to_numpy(),
                      "second": base_va["second"].to_numpy(), "split": "val", "label": y_va,
                      "p": p_va.astype(np.float64)}).write_parquet(os.path.join(PREDS, f"t1_se_seed{s}_val.parquet"))
        ok = bool(d_va["diff"] <= -0.0005 and d_va["ci95"][1] < 0)
        per_seed_pass.append(ok)
        diffs.append(d_va["diff"])
        res["per_seed"][s] = {"S_plus_eng_minus_S_validation": d_va, "S_plus_eng_minus_S_test_descriptive": d_te,
                              "pass": ok, "best_iteration": int(best)}
        print(f"seed {s}: validation S plus engagement minus S {d_va['diff']:+.6f} "
              f"CI [{d_va['ci95'][0]:+.6f}, {d_va['ci95'][1]:+.6f}], pass {ok}", flush=True)
        W.dump_json({"seeds_done": list(res["per_seed"]), "per_seed": res["per_seed"]}, OUT + ".partial")

    mean = float(np.mean(diffs))
    passes = bool(all(per_seed_pass) and mean <= -0.0005)
    res.update({"mean_over_seeds_validation": mean, "per_seed_pass": per_seed_pass, "passes": passes,
                "engagement_positive_rate": float(y_eng.mean()), "seconds": round(time.time() - t0)})
    W.dump_json(res, OUT)
    print(json.dumps({"mean_over_seeds_validation": mean, "passes": passes}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
