"""X10: calibration post-processing of the baseline XGBoost (GPU) win-probability model.

Trains the baseline params on CUDA, then fits on VALIDATION predictions:
  (a) one temperature on the logit
  (b) isotonic regression
  (c) per-time-bin temperature (seconds_since_round_live: 0-20, 20-40, 40-60, 60+)
Scores all on TEST with the helper's cluster bootstrap, paired against the uncalibrated GPU model.
"""
import gc
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")

import numpy as np
import pt_eval as E
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

T0 = time.time()
PRED = r"G:\pureskill\csds-cs2\data\viz\pt_preds"
os.makedirs(PRED, exist_ok=True)
MONO = (1, -1, 1, 0, 0, 0, 0, 1, 0, 0)
TIME_COL = 4  # seconds_since_round_live

D = E.load()
Xtr, ytr = D["train"]["X"], D["train"]["y"]
Xva, yva = D["val"]["X"], D["val"]["y"]
Xte, yte = D["test"]["X"], D["test"]["y"]
print("shapes train/val/test", Xtr.shape, Xva.shape, Xte.shape, flush=True)

params = dict(
    n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
    tree_method="hist", device="cuda", monotone_constraints=MONO, early_stopping_rounds=100,
    eval_metric="logloss", random_state=0, n_jobs=4,
)


def fit_model():
    m = xgb.XGBClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


model = None
for attempt in range(4):
    try:
        model = fit_model()
        break
    except Exception as e:  # retry only on GPU OOM, per task rules
        msg = str(e)
        print(f"attempt {attempt} failed: {msg[:300]}", flush=True)
        if "memory" not in msg.lower() and "oom" not in msg.lower() or attempt == 3:
            raise
        time.sleep(30)

bi = int(model.best_iteration)
print(f"GPU XGB best_iteration={bi} t={time.time()-T0:.1f}s", flush=True)
rng = (0, bi + 1)
pv_raw = model.predict_proba(Xva, iteration_range=rng)[:, 1].astype(np.float64)
pt_raw = model.predict_proba(Xte, iteration_range=rng)[:, 1].astype(np.float64)
np.save(os.path.join(PRED, "X10_val.npy"), pv_raw)
np.save(os.path.join(PRED, "X10_test.npy"), pt_raw)
# free training data and the booster: host RAM / commit is shared with other jobs
del model, Xtr, ytr
D.pop("train", None)
gc.collect()

# ---------- calibration helpers (numpy only) ----------
EPS = 1e-7


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_T(z, y):
    """1-D grid search on log-spaced temperatures, minimising mean log loss."""
    grid = np.exp(np.linspace(np.log(0.2), np.log(5.0), 3000))
    best_l, best_T = np.inf, 1.0
    for T in grid:
        l = float(E.ll_vec(sig(z / T), y).mean())
        if l < best_l:
            best_l, best_T = l, float(T)
    return best_T


def apply_T(p, T):
    return sig(logit(p) / T)


zv = logit(pv_raw)
zt = logit(pt_raw)
ll_raw_val = float(E.ll_vec(pv_raw, yva).mean())
print(f"raw val logloss {ll_raw_val:.5f}", flush=True)

# (a) single temperature
T_one = fit_T(zv, yva)
pv_a, pt_a = apply_T(pv_raw, T_one), apply_T(pt_raw, T_one)
print(f"(a) T={T_one:.4f} val ll {float(E.ll_vec(pv_a, yva).mean()):.5f}", flush=True)

# (b) isotonic
iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip").fit(pv_raw, yva)
pv_b, pt_b = iso.predict(pv_raw).astype(np.float64), iso.predict(pt_raw).astype(np.float64)
print(f"(b) isotonic val ll {float(E.ll_vec(pv_b, yva).mean()):.5f}", flush=True)

# (c) per-time-bin temperature
def tbin(x):
    return np.digitize(np.asarray(x, dtype=np.float64), [20.0, 40.0, 60.0])  # 0:<20 1:20-40 2:40-60 3:>=60


bv = tbin(Xva[:, TIME_COL])
bt = tbin(Xte[:, TIME_COL])
T_bin = {}
pv_c = np.empty_like(pv_raw)
pt_c = np.empty_like(pt_raw)
for k in range(4):
    mv = bv == k
    mt = bt == k
    T_bin[k] = fit_T(zv[mv], yva[mv]) if mv.sum() > 0 else 1.0
    pv_c[mv] = apply_T(pv_raw[mv], T_bin[k])
    pt_c[mt] = apply_T(pt_raw[mt], T_bin[k])
    print(f"(c) bin {k} n_val={int(mv.sum())} n_test={int(mt.sum())} T={T_bin[k]:.4f}", flush=True)
print(f"(c) per-bin val ll {float(E.ll_vec(pv_c, yva).mean()):.5f}", flush=True)

for nm, pv_, pt_ in (("X10_temp", pv_a, pt_a), ("X10_iso", pv_b, pt_b), ("X10_bintemp", pv_c, pt_c)):
    np.save(os.path.join(PRED, f"{nm}_val.npy"), pv_.astype(np.float64))
    np.save(os.path.join(PRED, f"{nm}_test.npy"), pt_.astype(np.float64))

# ---------- scoring ----------
# Test dict whose p_base is the uncalibrated GPU model: E.report then gives the paired diff vs GPU raw.
TG = dict(D["test"])
TG["p_base"] = pt_raw

print("=== REPORTS vs CPU baseline (helper default) ===", flush=True)
r_raw = E.report("X10_raw_gpu", D["test"], pt_raw)
r_a = E.report("X10_temp", D["test"], pt_a)
r_b = E.report("X10_iso", D["test"], pt_b)
r_c = E.report("X10_bintemp", D["test"], pt_c)

print("=== REPORTS paired vs uncalibrated GPU model (p_base = GPU raw) ===", flush=True)
g_a = E.report("X10_temp_vsgpu", TG, pt_a, save=False)
g_b = E.report("X10_iso_vsgpu", TG, pt_b, save=False)
g_c = E.report("X10_bintemp_vsgpu", TG, pt_c, save=False)

# ---------- diagnostics: middle band (raw 0.45-0.65) and per-bin reliability on test ----------
def band_gap(p, y, lo=0.45, hi=0.65):
    m = (pt_raw >= lo) & (pt_raw < hi)
    return float(np.mean(p[m]) - np.mean(y[m])), int(m.sum())


print("=== middle-band (raw 0.45-0.65 on test) mean(pred)-mean(y) ===", flush=True)
for nm, p in (("raw", pt_raw), ("temp", pt_a), ("iso", pt_b), ("bintemp", pt_c)):
    gap, n = band_gap(p, yte)
    print(f"  {nm:8s} gap={gap:+.4f} n={n}", flush=True)

print("=== per-time-bin test log loss raw vs bintemp ===", flush=True)
for k in range(4):
    m = bt == k
    if m.sum() == 0:
        continue
    lr = float(E.ll_vec(pt_raw[m], yte[m]).mean())
    lc = float(E.ll_vec(pt_c[m], yte[m]).mean())
    print(f"  bin {k} n={int(m.sum())} raw={lr:.5f} bintemp={lc:.5f} diff={lc-lr:+.5f}", flush=True)

summary = {
    "best_iteration": bi,
    "T_single": T_one,
    "T_bins": T_bin,
    "val_ll_raw": ll_raw_val,
    "test": {
        "raw_gpu": {"ll": r_raw["logloss"], "ci": r_raw["logloss_ci95"], "rw": r_raw["logloss_round_weighted"],
                    "brier": r_raw["brier"], "auc": r_raw["auc"], "ece": r_raw["ece10"],
                    "vs_cpu": r_raw["vs_cpu_xgb_baseline"], "vs_cpu_ci": r_raw["vs_cpu_xgb_baseline_ci95"]},
        "temp": {"ll": r_a["logloss"], "ci": r_a["logloss_ci95"], "rw": r_a["logloss_round_weighted"],
                 "brier": r_a["brier"], "auc": r_a["auc"], "ece": r_a["ece10"],
                 "vs_gpu": g_a["vs_cpu_xgb_baseline"], "vs_gpu_ci": g_a["vs_cpu_xgb_baseline_ci95"],
                 "vs_cpu": r_a["vs_cpu_xgb_baseline"]},
        "iso": {"ll": r_b["logloss"], "ci": r_b["logloss_ci95"], "rw": r_b["logloss_round_weighted"],
                "brier": r_b["brier"], "auc": r_b["auc"], "ece": r_b["ece10"],
                "vs_gpu": g_b["vs_cpu_xgb_baseline"], "vs_gpu_ci": g_b["vs_cpu_xgb_baseline_ci95"],
                "vs_cpu": r_b["vs_cpu_xgb_baseline"]},
        "bintemp": {"ll": r_c["logloss"], "ci": r_c["logloss_ci95"], "rw": r_c["logloss_round_weighted"],
                    "brier": r_c["brier"], "auc": r_c["auc"], "ece": r_c["ece10"],
                    "vs_gpu": g_c["vs_cpu_xgb_baseline"], "vs_gpu_ci": g_c["vs_cpu_xgb_baseline_ci95"],
                    "vs_cpu": r_c["vs_cpu_xgb_baseline"]},
    },
    "seconds": round(time.time() - T0, 1),
}
print("SUMMARY " + repr(summary), flush=True)
print(f"done in {time.time()-T0:.1f}s", flush=True)
