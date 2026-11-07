"""M3 extra: is the map effect confounded with the 2026-10-08 batch? (2026 kept set)

Recomputes the GroupKFold(5) out-of-fold predictions from M3_models.py (same settings, same
inner split) and saves them to scratchpad, then repeats the pooled map-vs-no-map comparison and
the map-label permutation test inside each batch separately:
  batch A = match_date 2026-10-08 (1,055 matches in the kept set)
  batch B = all other kept match dates (466 matches)
ASSUMPTION: match_date is the batch key here (the bake-off notes say most matches carry the
2026-10-08 date; whether this is the played date is UNVERIFIED in the bake-off notes).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M3_extra.py
"""

import importlib.util
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
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


bk = load_module("bakeoff_m3x", ROOT / "scripts" / "model" / "bakeoff.py")
FEATS = list(bk.FEATS)
MONO = list(bk.MONO)

S = pl.read_parquet(SCRATCH / "M3_states.parquet").sort("row_idx")
MT = pl.read_parquet(SCRATCH / "M3_matches.parquet")
X = S.select(FEATS).to_numpy().astype(np.float64)
y = S["label_ct_win"].to_numpy().astype(np.int64)
N = len(y)
mkeys = S["match_key"].to_list()
maps_list = S["map_name"].to_list()
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
batch_match = np.array(["A_2026-10-08" if match_date[m].startswith("2026-10-08") else "B_other"
                        for m in match_list])
state_batch = batch_match[mc]


def say(*a):
    print(*a, flush=True)


def xgb_model(mono):
    return xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4,
                             min_child_weight=20, subsample=0.8, tree_method="hist", device="cpu",
                             monotone_constraints=tuple(mono), early_stopping_rounds=100,
                             eval_metric="logloss", random_state=SEED)


def train_predict(Xf, train_mask, test_mask, mono):
    tr_codes = np.unique(mc[train_mask])
    ms = sorted(int(c) for c in tr_codes)
    rng = random.Random(SEED)
    rng.shuffle(ms)
    n_val = round(0.10 * len(ms))
    va_arr, inn_arr = np.array(sorted(ms[:n_val])), np.array(sorted(ms[n_val:]))
    m_in = train_mask & np.isin(mc, inn_arr)
    m_va = train_mask & np.isin(mc, va_arr)
    m = xgb_model(mono)
    m.fit(Xf[m_in], y[m_in], eval_set=[(Xf[m_va], y[m_va])], verbose=False)
    bi = int(m.best_iteration)
    return m.predict_proba(Xf[test_mask], iteration_range=(0, bi + 1))[:, 1], bi


def ll(yv, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(yv * np.log(p) + (1 - yv) * np.log(1 - p))


def boot_diff(la, lb, codes, seed=SEED):
    u, inv = np.unique(codes, return_inverse=True)
    n = np.bincount(inv).astype(float)
    sa = np.bincount(inv, weights=la, minlength=len(u))
    sb = np.bincount(inv, weights=lb, minlength=len(u))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    nn = n[idx].sum(1)
    d = sb[idx].sum(1) / nn - sa[idx].sum(1) / nn
    return {"a": float(sa.sum() / n.sum()), "b": float(sb.sum() / n.sum()),
            "diff": float(sb.sum() / n.sum() - sa.sum() / n.sum()),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "share_b_better": float((d < 0).mean()), "matches": int(len(u)), "states": int(n.sum())}


def perm_subset(resid, p, match_subset, seed=SEED):
    """Map-label permutation test restricted to matches in match_subset (bool over NM)."""
    R = np.bincount(mc, weights=resid, minlength=NM)[match_subset]
    Q = np.bincount(mc, weights=p * (1 - p), minlength=NM)[match_subset]
    mm = match_map[match_subset]
    J = len(map_names)

    def stat(mapping):
        num = np.bincount(mapping, weights=R, minlength=J) ** 2
        den = np.bincount(mapping, weights=Q, minlength=J)
        return float(np.sum(num[den > 0] / den[den > 0]))

    t_obs = stat(mm)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(mm)) for _ in range(B)])
    return {"stat": t_obs, "p_value": float((1 + np.sum(null >= t_obs)) / (B + 1)),
            "null_mean": float(null.mean()), "matches": int(match_subset.sum())}


# ---------------------------------------------------------------- recompute OOF (same settings)
say("== recompute GroupKFold(5) OOF (no map, map one-hot) ==")
X_map = np.hstack([X, OH])
MONO_MAP = MONO + [0] * OH.shape[1]
gkf = GroupKFold(n_splits=5)
fold_id = np.full(N, -1, dtype=np.int64)
p_nm = np.full(N, np.nan)
p_mp = np.full(N, np.nan)
for k, (tr_i, te_i) in enumerate(gkf.split(X, y, groups=mc)):
    fold_id[te_i] = k
    tr_m = np.zeros(N, dtype=bool)
    tr_m[tr_i] = True
    te_m = ~tr_m
    p_nm[te_m], _ = train_predict(X, tr_m, te_m, MONO)
    p_mp[te_m], _ = train_predict(X_map, tr_m, te_m, MONO_MAP)
np.savez(SCRATCH / "M3_oof.npz", p_nm=p_nm, p_mp=p_mp, fold_id=fold_id)
l_nm, l_mp = ll(y, p_nm), ll(y, p_mp)
say(f"pooled OOF: no map {l_nm.mean():.5f} (M3_models log 0.43769); map {l_mp.mean():.5f} "
    f"(M3_models log 0.43789)")

# ---------------------------------------------------------------- batch split
say("\n== within-batch comparison (GroupKFold OOF) ==")
results = {}
for b_name in ("A_2026-10-08", "B_other"):
    mb = state_batch == b_name
    d = boot_diff(l_nm[mb], l_mp[mb], mc[mb])
    match_sub = batch_match == b_name
    pt = perm_subset(y - p_nm, p_nm, match_sub)
    pt_map = perm_subset(y - p_mp, p_mp, match_sub)
    results[b_name] = {"pooled": d, "perm_nomap": pt, "perm_map": pt_map}
    say(f"{b_name}: matches {d['matches']}, states {d['states']}; no map {d['a']:.5f}, "
        f"map {d['b']:.5f}, diff {d['diff']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}], "
        f"share map better {d['share_b_better']:.3f}")
    say(f"    map-label permutation on no-map residuals: stat {pt['stat']:.2f} p {pt['p_value']:.4f} "
        f"(null mean {pt['null_mean']:.2f}); on map-model residuals: stat {pt_map['stat']:.2f} "
        f"p {pt_map['p_value']:.4f}")

say(f"\ndone in {time.time() - T0:.1f}s")
