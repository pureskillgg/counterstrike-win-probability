"""Per-second explanation of the THGNN-lite win-probability change (reverse-engineers the attribution layer of Szmida and
Toka, section 4, with a linear surrogate in place of their per-round ridge fit).

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\torch\\python.exe -I -X utf8 -u scripts\\model\\explain_thgnn_lite.py --map de_mirage

Steps, following the paper's attribution section:
1. Event rows: for each second t with a previous second in the same round, dW = p(t) - p(t-1) from the model.
2. Feature changes per slot between t-1 and t: alive, x, y, z, sin and cos of yaw, and the loadout channels. A change
   counts only while the slot is alive in both seconds, but the alive change itself is kept, so an elimination is
   attributed to the slot that died. This stands in for the paper's two-second post-elimination rule: dead slots carry
   no changes in our zero-filled arrays, so that rule reduces to this masking.
3. Surrogate: ridge regression (alpha 0.1, as in the paper) of dW on the standardised changes. The paper fits one ridge
   per round to reproduce that round's output. One global fit on a train sample is cheaper and is a deliberate change.
4. Linear SHAP: phi_j = w_j * (z_j - mean z_j). This is exact for a linear surrogate with independent features and the
   training mean as the baseline. It is checked against the surrogate output below.
5. Filter as in the paper: a feature unchanged over the interval gets 0.
6. Share (Eq 4.3): share_j = phi_j / sum_k |phi_k| * dW. Absolute shares add to |dW|. Signed shares add to dW only when
   the filtered phi of one row all share a sign, so the sign-consistent fraction is reported.
Writes data/temporal/<map>_thgnn_lite_explain.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_temporal_round_model as ttr  # noqa: E402  (build_table, previous_rows, split constants)

DATA = ttr.DATA
OUT = ttr.OUT
TRAIN, TEST = ttr.TRAIN, ttr.TEST
ALPHA = 0.1
FIT_SAMPLE = 400_000
N_SLOTS = 10
LOADOUT = ["health", "armor", "helmet", "defuser", "money", "equip_value", "defusing",
           "primary", "secondary", "flash", "smoke", "he_or_fire"]
CHANNELS = ["alive", "x", "y", "z", "sin_yaw", "cos_yaw"] + LOADOUT  # 18 per slot, slot-major in the flattened vector
N_CH = len(CHANNELS)
NAMES = [f"slot{j // N_CH}_{CHANNELS[j % N_CH]}" for j in range(N_SLOTS * N_CH)]


def change_features(rows, X, L, prev1):
    """Per-slot one-second changes for the given rows, flattened to (len(rows), 10 * 18)."""
    p = prev1[rows]
    has = p >= 0
    p0 = np.where(has, p, 0)
    cur = np.concatenate([X[rows], L[rows]], axis=2).astype(np.float32)
    old = np.concatenate([X[p0], L[p0]], axis=2).astype(np.float32)
    both_alive = (cur[:, :, :1] > 0.5) & (old[:, :, :1] > 0.5)
    d = (cur - old) * both_alive
    d[:, :, 0] = cur[:, :, 0] - old[:, :, 0]
    d[~has] = 0
    return d.reshape(rows.size, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="de_mirage")
    a = ap.parse_args()

    arr, X, table, label, state, usable = ttr.build_table(a.map)
    L = np.load(DATA / f"{a.map}_loadout.npy")
    pred = np.load(OUT / f"{a.map}_thgnn_lite_pred.npz")
    p = pred["p"].astype(np.float64)
    split = arr["split"].astype(np.int64)
    prev1 = ttr.previous_rows(arr["match_idx"], arr["round"], arr["second"], 2)[:, 1]
    ok = usable & (prev1 >= 0)

    rng = np.random.default_rng(0)
    fit_pool = np.flatnonzero(ok & (split == TRAIN))
    fit_rows = rng.choice(fit_pool, size=min(FIT_SAMPLE, fit_pool.size), replace=False)
    test_rows = np.flatnonzero(ok & (split == TEST))
    print(f"fit rows {fit_rows.size}, test rows {test_rows.size}", flush=True)

    # Surrogate: ridge on standardised changes, fitted on train only.
    dW_fit = p[fit_rows] - p[prev1[fit_rows]]
    D_fit = change_features(fit_rows, X, L, prev1).astype(np.float64)
    mu = D_fit.mean(axis=0)
    sd = D_fit.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Z_fit = (D_fit - mu) / sd
    ybar = dW_fit.mean()
    G = Z_fit.T @ Z_fit + ALPHA * np.eye(Z_fit.shape[1])
    w = np.linalg.solve(G, Z_fit.T @ (dW_fit - ybar))
    fit_r2 = 1 - np.sum((dW_fit - (Z_fit @ w + ybar)) ** 2) / np.sum((dW_fit - ybar) ** 2)
    del D_fit, Z_fit, G

    # Explain the test seconds.
    dW = p[test_rows] - p[prev1[test_rows]]
    D = change_features(test_rows, X, L, prev1).astype(np.float64)
    Z = (D - mu) / sd
    yhat = Z @ w + ybar
    phi = Z * w  # linear SHAP against the train-mean baseline (Z has zero mean on train)
    shap_check = float(np.max(np.abs(phi.sum(axis=1) + ybar - yhat)))
    S = phi * (D != 0)  # paper's filter: unchanged feature gets zero
    denom = np.abs(S).sum(axis=1)
    share = np.zeros_like(S)
    nz = denom > 0
    share[nz] = (S[nz] / denom[nz, None]) * dW[nz, None]

    ss_tot = np.sum((dW - dW.mean()) ** 2)
    r2 = float(1 - np.sum((dW - yhat) ** 2) / ss_tot)
    corr = float(np.corrcoef(dW, yhat)[0, 1])
    mixed = (S > 0).any(axis=1) & (S < 0).any(axis=1)
    sign_consistent = float(1 - mixed.mean())
    sums_match = np.abs(share.sum(axis=1) - dW) < 1e-6
    absS = np.abs(share)
    total = float(absS.sum())
    per_slot_channel = absS.reshape(-1, N_SLOTS, N_CH).sum(axis=0)
    per_channel = per_slot_channel.sum(axis=0)
    per_slot = per_slot_channel.sum(axis=1)
    top = np.argsort(-per_slot_channel.ravel())[:15]
    examples = []
    for i in np.argsort(-np.abs(dW))[:5]:
        r = test_rows[i]
        best = np.argsort(-np.abs(share[i]))[:6]
        examples.append({
            "match_key": str(arr["match_keys"][arr["match_idx"][r]]),
            "round": int(arr["round"][r]),
            "second": int(arr["second"][r]),
            "model_p": float(p[r]),
            "dW": float(dW[i]),
            "surrogate_dW": float(yhat[i]),
            "top": [{"feature": NAMES[j], "change": float(D[i, j]), "share": float(share[i, j])} for j in best],
        })

    result = {
        "map": a.map,
        "model": "THGNN-lite (data/temporal/%s_thgnn_lite_pred.npz)" % a.map,
        "surrogate": {"type": "global ridge on standardised per-slot changes", "alpha": ALPHA,
                      "fit_rows": int(fit_rows.size), "fit_split": "train", "fit_r2": float(fit_r2)},
        "test_rows": int(test_rows.size),
        "test_matches": int(np.unique(arr["match_idx"][test_rows]).size),
        "test_surrogate_r2_on_dW": r2,
        "test_surrogate_corr_on_dW": corr,
        "linear_shap_additivity_max_error": shap_check,
        "share_sums_to_dW_fraction": float(sums_match.mean()),
        "sign_consistent_fraction": sign_consistent,
        "model_dW_std": float(dW.std()),
        "share_of_absolute_attribution_by_side": {"CT_slots_0_4": float(per_slot[:5].sum() / total),
                                                  "T_slots_5_9": float(per_slot[5:].sum() / total)},
        "share_of_absolute_attribution_by_channel": {CHANNELS[i]: float(per_channel[i] / total) for i in range(N_CH)},
        "top_slot_channel": [{"feature": NAMES[j], "abs_share_total": float(per_slot_channel.ravel()[j] / total)}
                             for j in top],
        "examples": examples,
        "deviations_from_paper": [
            "global linear surrogate instead of one ridge per round",
            "1-second changes only; the paper also aggregates 1 to 5 second intervals",
            "player features are position, yaw, alive and loadout; the paper's feature list (weapon, speed, smoke, fire) "
            "is only partly available here",
            "no HLTV history features, consistent with the project's data rules",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.map}_thgnn_lite_explain.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("test_surrogate_r2_on_dW", "test_surrogate_corr_on_dW",
                                             "linear_shap_additivity_max_error", "sign_consistent_fraction")}),
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
