"""X7b: L2 logistic regression on a spline-and-interaction basis (numpy Newton, C = 1). CPU only.

Basis (column order of the raw feature matrix X in pt_eval.load):
  - one-hot of (alive_ct, alive_t) pair, observed in train (36 states), and the same one-hot x bomb_planted
  - bomb_planted main effect
  - piecewise-linear spline of seconds_since_round_live, knots 10,20,30,45,60,80 (x and hinge terms)
  - piecewise-linear spline of seconds_since_plant (clipped at 0, times bomb_planted), knots 10,20,30,45,60
  - ct_score_minus_t_score (linear), round_number > 12, and (not in spec, added) is_pistol_round, is_overtime
Standardised columns, unpenalised intercept, penalty 1/C per coefficient: same objective as
sklearn LogisticRegression(C=1.0). RidgeLogit is copied from scripts/model/per_tick_eval.py; the only change is
that the Newton Hessian is accumulated in row chunks and standardisation is done in place, to keep peak RAM
within the ~6 GB free on this machine.
Run with: <project CPU python> -I -X utf8 scripts\exp\X7b_lr_spline.py
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import sys
import time

import numpy as np

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")
import pt_eval as E  # noqa: E402

PRED_DIR = r"G:\pureskill\csds-cs2\data\viz\pt_preds"
ID = "X7b"
KNOTS_LIVE = [10, 20, 30, 45, 60, 80]
KNOTS_PLANT = [10, 20, 30, 45, 60]
K = 64  # pair code = alive_ct * K + alive_t
CHUNK = 200000


class RidgeLogit:
    """Standardised L2 logistic regression (C = 1, intercept unpenalised), Newton steps, numpy only.
    Copied from per_tick_eval.py. Column 0 of Z must be the intercept (ones); columns 1: are standardised."""

    def _standardise_fit(self, Z):
        n = Z.shape[0]
        s1 = np.zeros(Z.shape[1] - 1)
        s2 = np.zeros(Z.shape[1] - 1)
        for s in range(0, n, CHUNK):
            c = Z[s:s + CHUNK, 1:]
            s1 += c.sum(0)
            s2 += (c * c).sum(0)
        self.mu = s1 / n
        var = np.maximum(s2 / n - self.mu ** 2, 0.0)
        self.sd = np.sqrt(var) + 1e-12
        self._apply(Z)

    def _apply(self, Z):
        for s in range(0, Z.shape[0], CHUNK):
            Z[s:s + CHUNK, 1:] -= self.mu
            Z[s:s + CHUNK, 1:] /= self.sd

    def fit(self, Z, y, C=1.0, iters=30):
        self._standardise_fit(Z)
        n, f1 = Z.shape
        lam = np.full(f1, 1.0 / C)
        lam[0] = 0.0
        w = np.zeros(f1)
        for it in range(iters):
            eta = Z @ w
            p = 1.0 / (1.0 + np.exp(-eta))
            pc = np.clip(p, 1e-15, 1 - 1e-15)
            tr_ll = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
            g = Z.T @ (p - y) + lam * w
            H = np.zeros((f1, f1))
            for s in range(0, n, CHUNK):
                Zc = Z[s:s + CHUNK]
                wc = p[s:s + CHUNK] * (1.0 - p[s:s + CHUNK])
                H += (Zc * wc[:, None]).T @ Zc
            H += np.diag(lam)
            step = np.linalg.solve(H, g)
            w = w - step
            print(f"  newton it {it + 1:2d} train_logloss {tr_ll:.6f} max_step {np.abs(step).max():.3e}", flush=True)
            if np.abs(step).max() < 1e-9:
                break
        self.w = w
        return self

    def predict(self, Z):
        self._apply(Z)  # in place on a freshly built array
        return 1.0 / (1.0 + np.exp(-(Z @ self.w)))


def build(X, uniq):
    n = X.shape[0]
    ac = X[:, 0].astype(np.int64)
    at = X[:, 1].astype(np.int64)
    code = ac * K + at
    pos = np.minimum(np.searchsorted(uniq, code), len(uniq) - 1)
    hit = uniq[pos] == code
    P = len(uniq)
    planted = X[:, 3].astype(np.float64)
    sl = X[:, 4].astype(np.float64)
    sp = np.maximum(X[:, 5].astype(np.float64), 0.0) * planted
    live = [sl] + [np.maximum(sl - k, 0.0) for k in KNOTS_LIVE]
    plant = [c * planted for c in ([sp] + [np.maximum(sp - k, 0.0) for k in KNOTS_PLANT])]
    extra = [
        planted,
        X[:, 7].astype(np.float64),
        (X[:, 6] > 12).astype(np.float64),
        X[:, 8].astype(np.float64),
        X[:, 9].astype(np.float64),
    ]
    tail = live + plant + extra
    F = 1 + P + P + len(tail)
    Z = np.zeros((n, F))
    Z[:, 0] = 1.0
    rows = np.nonzero(hit)[0]
    Z[rows, 1 + pos[hit]] = 1.0
    Z[:, 1 + P:1 + 2 * P] = Z[:, 1:1 + P] * planted[:, None]
    for j, c in enumerate(tail):
        Z[:, 1 + 2 * P + j] = c
    return Z, int((~hit).sum())


t0 = time.time()
D = E.load()
Xtr = D["train"]["X"].astype(np.float64)
ytr = D["train"]["y"].astype(np.float64)
Xva = D["val"]["X"].astype(np.float64)
yva = D["val"]["y"].astype(np.int64)
Xte = D["test"]["X"].astype(np.float64)

uniq = np.unique(Xtr[:, 0].astype(np.int64) * K + Xtr[:, 1].astype(np.int64))
print("train alive-pair states observed:", len(uniq), flush=True)

Ztr, _ = build(Xtr, uniq)
del Xtr
print("design", Ztr.shape, round(time.time() - t0, 1), "s", flush=True)
model = RidgeLogit().fit(Ztr, ytr, C=1.0, iters=30)
del Ztr
print("fit done", round(time.time() - t0, 1), "s", flush=True)

Zva, unseen_va = build(Xva, uniq)
pv = model.predict(Zva).astype(np.float64)
del Zva
Zte, unseen_te = build(Xte, uniq)
pt = model.predict(Zte).astype(np.float64)
del Zte
print("val rows with unseen alive pair:", unseen_va, "test:", unseen_te, flush=True)
assert np.isfinite(pv).all() and np.isfinite(pt).all()

os.makedirs(PRED_DIR, exist_ok=True)
np.save(os.path.join(PRED_DIR, f"{ID}_val.npy"), pv)
np.save(os.path.join(PRED_DIR, f"{ID}_test.npy"), pt)
val_ll = float(E.ll_vec(pv, yva).mean())
print("val logloss", round(val_ll, 6), flush=True)

E.report(
    ID,
    D["test"],
    pt,
    extra={
        "model": "L2 logistic (C=1) on spline+interaction basis, numpy Newton, 91 columns",
        "device": "cpu (numpy)",
        "val_logloss": val_ll,
        "unseen_pair_rows_test": unseen_te,
    },
)
print("total s", round(time.time() - t0, 1), flush=True)
