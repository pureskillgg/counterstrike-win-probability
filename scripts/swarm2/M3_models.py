"""M3 models: is map_name worth a model feature? 2026 kept set (1,521 matches, 234,265 states).

Inputs (scratchpad, written by scripts/swarm2/M3_build.py):
  M3_states.parquet, M3_matches.parquet
Model: XGBoost with the bake-off settings (scripts/model/bakeoff.py, CPU, seed 0):
  n_estimators 3000, eta 0.05, depth 4, min_child_weight 20, subsample 0.8, hist, monotone
  constraints as the bake-off (0 for map columns), early stopping 100 on a validation set.
ASSUMPTIONS (marked):
  A1. Early stopping inside each training fold uses a match-level inner split of 10% of the
      training matches (mirrors the bake-off's validation share). The outer test fold is never
      used for stopping.
  A2. map_name is taken from the header tome as the played map (only map source in the tome).
  A3. One-hot map columns: all 10 kept maps, no drop (trees do not need a reference level).
  A4. Leave-one-map-out (LOMO): the held-out map's one-hot column is removed from the design, so
      its rows are all-zero. This is the model's routing for an unseen category, not a learned
      estimate for that map.
  A5. Bootstrap: match-cluster bootstrap, 2,000 resamples, seed 0, percentile intervals.
  A6. Per-map offset (cross-fitted): one logit shift per map, fitted on the other GroupKFold
      folds' no-map out-of-fold logits. The out-of-fold logits of the other folds came from models
      that saw the held-out fold, so this is approximate.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M3_models.py
"""

import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SEED = 0
B = 2000
EPS = 1e-15
T0 = time.time()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bk = load_module("bakeoff_m3", ROOT / "scripts" / "model" / "bakeoff.py")
FEATS = list(bk.FEATS)
MONO = list(bk.MONO)

S = pl.read_parquet(SCRATCH / "M3_states.parquet").sort("row_idx")
MT = pl.read_parquet(SCRATCH / "M3_matches.parquet")

X = S.select(FEATS).to_numpy().astype(np.float64)
y = S["label_ct_win"].to_numpy().astype(np.int64)
N = len(y)
mkeys = S["match_key"].to_list()
maps_list = S["map_name"].to_list()
split_p = np.array(S["split_primary"].to_list())

match_list = sorted(set(mkeys))
mcode = {m: i for i, m in enumerate(match_list)}
mc = np.array([mcode[m] for m in mkeys], dtype=np.int64)
NM = len(match_list)

map_names = sorted(set(maps_list))
map_idx = {mp: i for i, mp in enumerate(map_names)}
ms_code = np.array([map_idx[m] for m in maps_list], dtype=np.int64)
OH = np.zeros((N, len(map_names)))
OH[np.arange(N), ms_code] = 1.0
match_map = np.zeros(NM, dtype=np.int64)
match_map[mc] = ms_code
match_date = dict(zip(MT["match_key"].to_list(), MT["match_date"].to_list()))
batch_match = np.array(["2026-10-08" if match_date[m].startswith("2026-10-08") else "other"
                        for m in match_list])
state_batch = batch_match[mc]
n_matches_by_map = np.bincount(match_map, minlength=len(map_names))
n_states_by_map = np.bincount(ms_code, minlength=len(map_names))
ct_by_map = np.bincount(ms_code, weights=y.astype(float), minlength=len(map_names)) / n_states_by_map
top8 = sorted(range(len(map_names)), key=lambda j: -n_matches_by_map[j])[:8]

RESULTS = {"n_states": N, "n_matches": NM, "maps": {}}


def say(*a):
    print(*a, flush=True)


def xgb_model(mono):
    return xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4,
                             min_child_weight=20, subsample=0.8, tree_method="hist", device="cpu",
                             monotone_constraints=tuple(mono), early_stopping_rounds=100,
                             eval_metric="logloss", random_state=SEED)


def fit_predict(Xtr, ytr, Xva, yva, Xte, mono):
    m = xgb_model(mono)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    bi = int(m.best_iteration)
    return m.predict_proba(Xte, iteration_range=(0, bi + 1))[:, 1], bi


def inner_split(match_codes):
    ms = sorted(int(c) for c in match_codes)
    rng = random.Random(SEED)
    rng.shuffle(ms)
    n_val = round(0.10 * len(ms))
    return set(ms[:n_val]), set(ms[n_val:])


def train_predict(Xf, train_mask, test_mask, mono):
    tr_codes = np.unique(mc[train_mask])
    va, inn = inner_split(tr_codes)
    va_arr, inn_arr = np.array(sorted(va)), np.array(sorted(inn))
    m_in = train_mask & np.isin(mc, inn_arr)
    m_va = train_mask & np.isin(mc, va_arr)
    p, bi = fit_predict(Xf[m_in], y[m_in], Xf[m_va], y[m_va], Xf[test_mask], mono)
    info = {"best_iteration": bi, "inner_val_matches": len(va), "inner_train_matches": len(inn),
            "inner_train_states": int(m_in.sum()), "inner_val_states": int(m_va.sum())}
    return p, bi, info


def ll(yv, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(yv * np.log(p) + (1 - yv) * np.log(1 - p))


def pct(a, q):
    return float(np.percentile(a, q))


def boot_diff(la, lb, codes, seed=SEED):
    """Pooled log-loss difference lb - la with a match-cluster bootstrap (paired)."""
    u, inv = np.unique(codes, return_inverse=True)
    n = np.bincount(inv).astype(float)
    sa = np.bincount(inv, weights=la, minlength=len(u))
    sb = np.bincount(inv, weights=lb, minlength=len(u))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    nn = n[idx].sum(1)
    d = sb[idx].sum(1) / nn - sa[idx].sum(1) / nn
    pt_a, pt_b = sa.sum() / n.sum(), sb.sum() / n.sum()
    return {"a": float(pt_a), "b": float(pt_b), "diff": float(pt_b - pt_a), "lo": pct(d, 2.5),
            "hi": pct(d, 97.5), "share_b_better": float((d < 0).mean()), "matches": int(len(u)),
            "states": int(n.sum())}


def boot_mean(vals, codes, seed=SEED):
    """Mean of vals with a match-cluster bootstrap."""
    u, inv = np.unique(codes, return_inverse=True)
    n = np.bincount(inv).astype(float)
    s = np.bincount(inv, weights=vals, minlength=len(u))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    bm = s[idx].sum(1) / n[idx].sum(1)
    return {"mean": float(s.sum() / n.sum()), "lo": pct(bm, 2.5), "hi": pct(bm, 97.5),
            "matches": int(len(u)), "states": int(n.sum())}


def fit_offsets(z, yv, mcodes, J, iters=30):
    b = np.zeros(J)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(z + b[mcodes])))
        g = np.bincount(mcodes, weights=(yv - p), minlength=J)
        h = np.bincount(mcodes, weights=p * (1 - p), minlength=J)
        b += np.where(h > 1e-9, g / np.maximum(h, 1e-9), 0.0)
    return b


def perm_map_test(resid, p, seed=SEED):
    """Match-level permutation test of a map effect in residuals (labels permuted across matches)."""
    R = np.bincount(mc, weights=resid, minlength=NM)
    Q = np.bincount(mc, weights=p * (1 - p), minlength=NM)
    J = len(map_names)

    def stat(mapping):
        num = np.bincount(mapping, weights=R, minlength=J) ** 2
        den = np.bincount(mapping, weights=Q, minlength=J)
        return float(np.sum(num[den > 0] / den[den > 0]))

    t_obs = stat(match_map)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(match_map)) for _ in range(B)])
    return {"stat": t_obs, "p_value": float((1 + np.sum(null >= t_obs)) / (B + 1)),
            "null_mean": float(null.mean()), "null_p95": pct(null, 95)}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- 0. data check
say("== 0. Data ==")
say(f"states {N}; matches {NM}; maps {len(map_names)}; features {FEATS}")
say("matches by map (kept set):")
for j in sorted(range(len(map_names)), key=lambda j: -n_matches_by_map[j]):
    say(f"  {map_names[j]:<12} matches {n_matches_by_map[j]:>4}  states {n_states_by_map[j]:>6}"
        f"  CT share (states) {ct_by_map[j]:.4f}")
say(f"top-8 by matches (LOMO set): {[map_names[j] for j in top8]}")
say(f"batch 2026-10-08 matches: {(batch_match == '2026-10-08').sum()} of {NM}")

# ---------------------------------------------------------------- 1. reproduction
say("\n== 1. Reproduction of the bake-off XGBoost (primary split, 10 features, no map) ==")
tr_p, va_p, te_p = split_p == "train", split_p == "val", split_p == "test"
p_rep, bi_rep = fit_predict(X[tr_p], y[tr_p], X[va_p], y[va_p], X[te_p], MONO)
ll_rep = float(log_loss(y[te_p], np.clip(p_rep, 0, 1), labels=[0, 1]))
say(f"best_iteration {bi_rep} (bake-off log: 261); primary test log loss {ll_rep:.5f} "
    f"(bake-off log: 0.43660)")
RESULTS["reproduction"] = {"best_iteration": bi_rep, "test_logloss": ll_rep,
                           "bakeoff_logloss": 0.43660, "bakeoff_best_iteration": 261}

# ---------------------------------------------------------------- 2. GroupKFold(5)
say("\n== 2. GroupKFold(5) by match: XGBoost without vs with map_name one-hot ==")
X_map = np.hstack([X, OH])
MONO_MAP = MONO + [0] * OH.shape[1]
gkf = GroupKFold(n_splits=5)
fold_id = np.full(N, -1, dtype=np.int64)
p_nm = np.full(N, np.nan)
p_mp = np.full(N, np.nan)
folds = []
for k, (tr_i, te_i) in enumerate(gkf.split(X, y, groups=mc)):
    fold_id[te_i] = k
    tr_m = np.zeros(N, dtype=bool)
    tr_m[tr_i] = True
    te_m = ~tr_m
    p1, bi1, info1 = train_predict(X, tr_m, te_m, MONO)
    p_nm[te_m] = p1
    p2, bi2, info2 = train_predict(X_map, tr_m, te_m, MONO_MAP)
    p_mp[te_m] = p2
    say(f"  fold {k}: test matches {len(np.unique(mc[te_m]))}, states {int(te_m.sum())}; "
        f"no-map best_iter {bi1} (inner val {info1['inner_val_matches']} matches); "
        f"map best_iter {bi2}")
    folds.append({"fold": k, "test_matches": int(len(np.unique(mc[te_m]))),
                  "test_states": int(te_m.sum()), "best_iter_nomap": bi1, "best_iter_map": bi2,
                  "inner_val_matches": info1["inner_val_matches"]})
assert not np.isnan(p_nm).any() and not np.isnan(p_mp).any()
l_nm, l_mp = ll(y, p_nm), ll(y, p_mp)
pooled = boot_diff(l_nm, l_mp, mc)
say(f"pooled OOF log loss: no map {pooled['a']:.5f}; with map {pooled['b']:.5f}; "
    f"difference (map minus no map) {pooled['diff']:+.5f}")
say(f"  match-cluster bootstrap 95% CI of the difference: [{pooled['lo']:+.5f}, {pooled['hi']:+.5f}]; "
    f"share of resamples where the map model is better: {pooled['share_b_better']:.3f}")
fold_rows = []
for k in range(5):
    m_k = fold_id == k
    fold_rows.append({"fold": k, "nomap": float(l_nm[m_k].mean()), "map": float(l_mp[m_k].mean()),
                      "diff": float(l_mp[m_k].mean() - l_nm[m_k].mean())})
    say(f"  fold {k}: no map {fold_rows[-1]['nomap']:.5f}, map {fold_rows[-1]['map']:.5f}, "
        f"diff {fold_rows[-1]['diff']:+.5f}")
say(f"  folds where map helps: {sum(1 for r in fold_rows if r['diff'] < 0)} of 5")
RESULTS["groupkfold"] = {"pooled": pooled, "folds": fold_rows, "fold_detail": folds}

# ---------------------------------------------------------------- 3. per-map OOF and residuals
say("\n== 3. Per-map out-of-fold results (GroupKFold) and residuals ==")
r_nm = y - p_nm
r_mp = y - p_mp
per_map = []
for j in sorted(range(len(map_names)), key=lambda j: -n_matches_by_map[j]):
    m_j = ms_code == j
    d = boot_diff(l_nm[m_j], l_mp[m_j], mc[m_j])
    rn = boot_mean(r_nm[m_j], mc[m_j])
    rm = boot_mean(r_mp[m_j], mc[m_j])
    b1 = state_batch[m_j] == "2026-10-08"
    rb1 = float(r_nm[m_j][b1].mean()) if b1.any() else float("nan")
    rb0 = float(r_nm[m_j][~b1].mean()) if (~b1).any() else float("nan")
    row = {"map": map_names[j], "matches": int(n_matches_by_map[j]), "states": int(n_states_by_map[j]),
           "ct_share": float(ct_by_map[j]), "ll_nomap": d["a"], "ll_map": d["b"],
           "diff_map_minus_nomap": d["diff"], "diff_lo": d["lo"], "diff_hi": d["hi"],
           "resid_nomap": rn["mean"], "resid_nomap_lo": rn["lo"], "resid_nomap_hi": rn["hi"],
           "resid_map": rm["mean"], "resid_map_lo": rm["lo"], "resid_map_hi": rm["hi"],
           "resid_nomap_batch_1008": rb1, "resid_nomap_batch_other": rb0,
           "states_batch_1008": int(b1.sum())}
    per_map.append(row)
    say(f"  {map_names[j]:<11} m={row['matches']:>4} st={row['states']:>6} CT={row['ct_share']:.3f} "
        f"LL nomap {row['ll_nomap']:.5f} map {row['ll_map']:.5f} diff {row['diff_map_minus_nomap']:+.5f} "
        f"[{row['diff_lo']:+.5f},{row['diff_hi']:+.5f}]")
    say(f"      resid(y-p) nomap {row['resid_nomap']:+.4f} [{row['resid_nomap_lo']:+.4f},"
        f"{row['resid_nomap_hi']:+.4f}]; map {row['resid_map']:+.4f} "
        f"[{row['resid_map_lo']:+.4f},{row['resid_map_hi']:+.4f}]; "
        f"nomap resid by batch 10-08 {rb1:+.4f} (n={int(b1.sum())}) vs other {rb0:+.4f}")
RESULTS["per_map_oof"] = per_map

perm_nm = perm_map_test(r_nm, p_nm)
perm_mp = perm_map_test(r_mp, p_mp)
say(f"\nmap-label permutation test (match-level, {B} permutations) on residuals:")
say(f"  no-map OOF residuals: stat {perm_nm['stat']:.2f}, p {perm_nm['p_value']:.4f} "
    f"(null mean {perm_nm['null_mean']:.2f}, 95th {perm_nm['null_p95']:.2f})")
say(f"  map OOF residuals:    stat {perm_mp['stat']:.2f}, p {perm_mp['p_value']:.4f} "
    f"(null mean {perm_mp['null_mean']:.2f}, 95th {perm_mp['null_p95']:.2f})")
RESULTS["perm_test"] = {"nomap": perm_nm, "map": perm_mp}

# ---------------------------------------------------------------- 3b. per-map priors (offset)
say("\n== 4. Per-map priors: cross-fitted logit offset on no-map OOF (approximate, see A6) ==")
z_nm = logit(p_nm)
p_off = np.full(N, np.nan)
offsets = []
for k in range(5):
    te_m = fold_id == k
    tr_m = ~te_m
    b = fit_offsets(z_nm[tr_m], y[tr_m], ms_code[tr_m], len(map_names))
    p_off[te_m] = 1.0 / (1.0 + np.exp(-(z_nm[te_m] + b[ms_code[te_m]])))
    offsets.append(b)
l_off = ll(y, p_off)
d_off = boot_diff(l_nm, l_off, mc)
say(f"pooled OOF log loss: no map {d_off['a']:.5f}; per-map offset {d_off['b']:.5f}; "
    f"difference {d_off['diff']:+.5f} [{d_off['lo']:+.5f}, {d_off['hi']:+.5f}]; "
    f"share of resamples better {d_off['share_b_better']:.3f}")
mean_off = np.mean(np.vstack(offsets), axis=0)
say("mean offset (logit) by map, averaged over folds: " +
    ", ".join(f"{map_names[j]} {mean_off[j]:+.3f}" for j in sorted(range(len(map_names)),
                                                              key=lambda j: -n_matches_by_map[j])))
RESULTS["per_map_offset"] = {"pooled": d_off, "mean_offset_logit": {map_names[j]: float(mean_off[j])
                                                                    for j in range(len(map_names))}}

# ---------------------------------------------------------------- 5. LOMO for the 8 largest maps
say("\n== 5. Leave-one-map-out, eight largest maps (by kept matches) ==")
lomo_rows = []
for j in top8:
    te_m = ms_code == j
    tr_m = ~te_m
    keep = [c for c in range(len(map_names)) if c != j]
    X_lomo = np.hstack([X, OH[:, keep]])
    MONO_LOMO = MONO + [0] * len(keep)
    pl_nm, bi_a, info_a = train_predict(X, tr_m, te_m, MONO)
    pl_mp, bi_b, info_b = train_predict(X_lomo, tr_m, te_m, MONO_LOMO)
    la_oof = l_nm[te_m]
    ll_l_nm = ll(y[te_m], pl_nm)
    ll_l_mp = ll(y[te_m], pl_mp)
    d_lo_oof = boot_diff(la_oof, ll_l_nm, mc[te_m])
    d_unseen = boot_diff(ll_l_nm, ll_l_mp, mc[te_m])
    rl = boot_mean(y[te_m] - pl_nm, mc[te_m])
    row = {"map": map_names[j], "matches": int(n_matches_by_map[j]), "states": int(te_m.sum()),
           "ll_oof_nomap_same_states": d_lo_oof["a"], "ll_lomo_nomap": d_lo_oof["b"],
           "diff_lomo_minus_oof": d_lo_oof["diff"], "diff_lo": d_lo_oof["lo"], "diff_hi": d_lo_oof["hi"],
           "ll_lomo_map_unseen": d_unseen["b"], "diff_unseen_minus_nomap": d_unseen["diff"],
           "unseen_lo": d_unseen["lo"], "unseen_hi": d_unseen["hi"],
           "resid_lomo_nomap": rl["mean"], "resid_lo": rl["lo"], "resid_hi": rl["hi"],
           "best_iter_nomap": bi_a, "best_iter_map": bi_b,
           "inner_val_matches_nomap": info_a["inner_val_matches"]}
    lomo_rows.append(row)
    say(f"  {map_names[j]:<11} m={row['matches']:>4} st={row['states']:>5}  OOF nomap "
        f"{row['ll_oof_nomap_same_states']:.5f} | LOMO nomap {row['ll_lomo_nomap']:.5f} "
        f"(diff {row['diff_lomo_minus_oof']:+.5f} [{row['diff_lo']:+.5f},{row['diff_hi']:+.5f}]) | "
        f"LOMO map-unseen {row['ll_lomo_map_unseen']:.5f} (vs nomap {row['diff_unseen_minus_nomap']:+.5f} "
        f"[{row['unseen_lo']:+.5f},{row['unseen_hi']:+.5f}])")
    say(f"      LOMO residual (y-p) {row['resid_lomo_nomap']:+.4f} [{row['resid_lo']:+.4f},"
        f"{row['resid_hi']:+.4f}]; best iter nomap {bi_a} map {bi_b}")
RESULTS["lomo_top8"] = lomo_rows

with open(SCRATCH / "M3_results.json", "w", encoding="utf-8") as fh:
    json.dump(RESULTS, fh, indent=1, default=float)
say(f"\nwrote {SCRATCH / 'M3_results.json'}")
say(f"done in {time.time() - T0:.1f}s")
