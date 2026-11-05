"""Explanations for single predictions of the 83-input XGBoost model, contrasts between two predictions, and interactions.

This is local explanation, not feature importance. For one second of a round it asks where the prediction came from:
  absolute   each input's contribution against the model's base value (waterfall)
  relative   the difference between two predictions, input by input (contrast)
  interact   which pairs of inputs contribute jointly, beyond their separate contributions (SHAP interaction values)

Nothing is retrained. The script reads the artifacts that `xgb83_explain.py --stage explain` wrote to WORK (booster,
shap_sample.npz, xs.npy, train_meta.json) and the validation key table that `experiments/sprint/sprint_prepare.py` wrote
into the matrix directory (val_keys.parquet). Run the explain stage first.

Run:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe scripts\\explain\\xgb83_local.py

Outputs under docs/explainability/local/:
  examples.csv                  four chosen seconds: prediction, label, match, round, second
  waterfall_<example>.csv       one second: each input's logit contribution, largest first, with its value and percentile
  contrast_<pair>.csv           two seconds: each input's logit difference (A minus B), largest first
  interactions_top_pairs.csv    strongest input pairs by mean absolute SHAP interaction value
  local_explain_meta.json       settings, the choice rules, and the consistency checks

Units. Contributions are on the logit scale, which is additive: base + sum of contributions = z, and P(CT wins) = sigmoid(z).
The probability column in the waterfall is the leave-one-out effect 100*(sigmoid(z) - sigmoid(z - phi)) in percentage
points. Those effects do not add up to the prediction's change, because the sigmoid is not linear.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgb83_explain import DOCS, GROUPS, MDIR, WORK, sig  # noqa: E402

OUT = os.path.join(DOCS, "local")
VAL_KEYS = os.path.join(MDIR, "val_keys.parquet")
RNG_SEED = 20261105
INTERACTION_N = 2000  # seconds from the SHAP sample for the interaction values
NEIGHBOUR_FEATURES = 10  # inputs used to find the nearest similar second
SAME_STATE = ["alive_ct", "alive_t", "bomb_planted"]  # the neighbour must match these exactly
EPS = 1e-15


def logloss_rows(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(MDIR, "meta.json")))
    feats = meta["feats_all"]
    S = np.load(os.path.join(WORK, "shap_sample.npz"))
    phi = S["phi"].astype(np.float64)  # (n, 83) TreeSHAP contributions on the logit scale
    bias = S["bias"].astype(np.float64)
    z = S["z"].astype(np.float64)
    idx = S["idx"]  # positions in the validation rows
    xs = np.load(os.path.join(WORK, "xs.npy")).astype(np.float64)
    n = phi.shape[0]
    assert phi.shape == (n, len(feats)) and xs.shape == (n, len(feats)), "sample arrays do not match the 83 inputs"
    assert np.allclose(bias, bias[0]), "base value differs between rows"
    base = float(bias[0])
    p = sig(z)

    vk = pl.read_parquet(VAL_KEYS)
    assert vk.height == meta["n_val"], "val_keys rows differ from the validation rows"
    for c in ["match_key", "round", "second", "label_ct_win"]:
        assert c in vk.columns, f"val_keys lacks {c}"
    mk = vk["match_key"].to_list()
    rnd = vk["round"].to_list()
    sec = vk["second"].to_list()
    label = vk["label_ct_win"].to_numpy().astype(np.float64)[idx]
    key = [(mk[i], rnd[i], sec[i]) for i in idx]

    grp = {f: g for g, cols in GROUPS.items() for f in cols}
    j_of = {f: j for j, f in enumerate(feats)}
    sorted_cols = []
    for j in range(len(feats)):
        col = xs[:, j]
        sorted_cols.append(np.sort(col[~np.isnan(col)]))

    def percentile(j, v):
        if v is None or np.isnan(v) or len(sorted_cols[j]) == 0:
            return None
        return float(100.0 * np.searchsorted(sorted_cols[j], v, side="right") / len(sorted_cols[j]))

    def value(k, j):
        v = xs[k, j]
        return None if np.isnan(v) else float(v)

    # ---- choose four seconds (rules are fixed; recorded in the meta file) ----
    ll = logloss_rows(p, label)
    k_loss = int(np.argmax(ll))
    ct_idx = np.where(label == 1)[0]
    t_idx = np.where(label == 0)[0]
    k_ct = int(ct_idx[np.argmax(p[ct_idx])])
    k_t = int(t_idx[np.argmin(p[t_idx])])
    k_close = int(np.argmin(np.abs(p - 0.5)))
    examples = {
        "largest_loss": (k_loss, "the second with the largest log loss in the sample"),
        "confident_ct_win": (k_ct, "label CT wins; the highest P(CT wins) among those seconds"),
        "confident_t_win": (k_t, "label T wins; the lowest P(CT wins) among those seconds"),
        "close_call": (k_close, "P(CT wins) closest to 0.5"),
    }

    ex_rows = []
    for name, (k, rule) in examples.items():
        ex_rows.append({"example": name, "rule": rule, "sample_position": k, "match_key": str(key[k][0]),
                        "round": int(key[k][1]), "second": int(key[k][2]), "label_ct_win": int(label[k]),
                        "p_ct_win": float(p[k]), "logit_z": float(z[k]), "base_logit": base,
                        "log_loss_row": float(ll[k])})
    pl.DataFrame(ex_rows).write_csv(os.path.join(OUT, "examples.csv"))

    # ---- absolute: waterfall for each example ----
    for name, (k, _) in examples.items():
        order = np.argsort(-np.abs(phi[k]))
        rows = []
        for rank, j in enumerate(order):
            f = feats[j]
            rows.append({"rank": rank + 1, "feature": f, "bundle": grp[f], "value": value(k, j),
                         "percentile_in_sample": percentile(j, value(k, j)),
                         "logit_contribution": float(phi[k, j]),
                         "pp_leave_out": float(100.0 * (p[k] - sig(z[k] - phi[k, j])))})
        pl.DataFrame(rows).write_csv(os.path.join(OUT, f"waterfall_{name}.csv"))
        # consistency: base + sum of contributions equals z
        assert abs(base + phi[k].sum() - z[k]) < 1e-3, "waterfall does not add up to z"

    # ---- relative: contrasts between two seconds ----
    def contrast(a, b, name):
        d = phi[a] - phi[b]
        order = np.argsort(-np.abs(d))
        rows = []
        for rank, j in enumerate(order):
            f = feats[j]
            rows.append({"rank": rank + 1, "feature": f, "bundle": grp[f], "value_a": value(a, j), "value_b": value(b, j),
                         "percentile_a": percentile(j, value(a, j)), "percentile_b": percentile(j, value(b, j)),
                         "logit_a": float(phi[a, j]), "logit_b": float(phi[b, j]), "logit_a_minus_b": float(d[j])})
        pl.DataFrame(rows).write_csv(os.path.join(OUT, f"contrast_{name}.csv"))
        gap = float(z[a] - z[b])
        assert abs(d.sum() - gap) < 1e-3, "contrast does not add up to the logit gap"
        return {"a": int(a), "b": int(b), "z_a": float(z[a]), "z_b": float(z[b]), "p_a": float(p[a]), "p_b": float(p[b]),
                "logit_gap": gap, "sum_of_input_differences": float(d.sum())}

    contrasts = {}
    contrasts["confident_ct_vs_t"] = contrast(k_ct, k_t, "confident_ct_vs_t")

    # nearest similar second to the largest-loss row: same alive counts and bomb state, similar top inputs, far apart in P
    top_j = np.argsort(-np.abs(phi).mean(axis=0))[:NEIGHBOUR_FEATURES]
    cols = xs[:, top_j]
    mu = np.nanmean(cols, axis=0)
    sd = np.nanstd(cols, axis=0)
    sd[sd == 0] = 1.0
    Zc = np.nan_to_num((cols - mu) / sd, nan=0.0)
    same = np.ones(n, dtype=bool)
    for f in SAME_STATE:
        same &= xs[:, j_of[f]] == xs[k_loss, j_of[f]]
    # calibration check for the anchor's state: does the model's mean P match the observed CT-win rate there?
    group = {"alive_ct": float(xs[k_loss, j_of["alive_ct"]]), "alive_t": float(xs[k_loss, j_of["alive_t"]]),
             "bomb_planted": float(xs[k_loss, j_of["bomb_planted"]]), "seconds": int(same.sum()),
             "mean_predicted_p_ct_win": float(p[same].mean()), "observed_ct_win_rate": float(label[same].mean())}

    rule_used = None
    for rule_name, mask in [
        ("same state, P differs by at least 0.25", same & (np.abs(p - p[k_loss]) >= 0.25)),
        ("same state, P differs by at least 0.10", same & (np.abs(p - p[k_loss]) >= 0.10)),
        ("same state, any P", same),
    ]:
        mask = mask.copy()
        mask[k_loss] = False
        if mask.any():
            rule_used = rule_name
            break
    if rule_used is None:  # no second with the same state: fall back to any other second
        mask = np.ones(n, dtype=bool)
        mask[k_loss] = False
        rule_used = "any second (no same-state candidate)"
    cand = np.where(mask)[0]
    dist = np.sqrt(((Zc[cand] - Zc[k_loss]) ** 2).sum(axis=1))
    k_nb = int(cand[np.argmin(dist)])
    contrasts["largest_loss_vs_neighbour"] = contrast(k_loss, k_nb, "largest_loss_vs_neighbour")
    contrasts["largest_loss_vs_neighbour"]["neighbour_rule"] = rule_used
    contrasts["largest_loss_vs_neighbour"]["neighbour_candidates"] = int(len(cand))

    # ---- interact: SHAP interaction values on a subsample ----
    bst = xgb.Booster()
    bst.load_model(os.path.join(WORK, "xgb83_s0.json"))
    best = int(json.load(open(os.path.join(WORK, "train_meta.json")))["best_iteration"])
    rng = np.random.default_rng(RNG_SEED)
    m = min(INTERACTION_N, n)
    sel = np.sort(rng.choice(n, size=m, replace=False))
    t1 = time.time()
    M = bst.predict(xgb.DMatrix(xs[sel].astype(np.float32)), pred_interactions=True, iteration_range=(0, best + 1))
    M = np.asarray(M, dtype=np.float64)
    t_inter = time.time() - t1
    row_sum_err = float(np.abs(M[:, :83, :83].sum(axis=2) - phi[sel]).max())
    assert row_sum_err < 1e-3, f"interaction rows do not sum to SHAP values ({row_sum_err})"
    iu, ju = np.triu_indices(83, 1)
    vals = M[:, iu, ju]
    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(-mean_abs)[:20]
    mabs = np.abs(phi[sel]).mean(axis=0)  # context: typical size of each input's own contribution
    pairs = []
    for r in order:
        a, b = feats[iu[r]], feats[ju[r]]
        pairs.append({"rank": len(pairs) + 1, "feature_a": a, "bundle_a": grp[a], "feature_b": b, "bundle_b": grp[b],
                      "mean_abs_interaction_logit": float(mean_abs[r]),
                      "pair_total_logit_standard": float(2 * mean_abs[r]),
                      "mean_abs_shap_a_logit": float(mabs[iu[r]]), "mean_abs_shap_b_logit": float(mabs[ju[r]])})
    pl.DataFrame(pairs).write_csv(os.path.join(OUT, "interactions_top_pairs.csv"))

    out_meta = {
        "booster": "xgb83_s0.json", "best_iteration": best, "n_sample": int(n),
        "base_logit": base, "base_probability": float(sig(base)),
        "examples": {k: {"rule": r, "sample_position": int(k_), "p_ct_win": float(p[k_]), "label_ct_win": int(label[k_])}
                     for k, (k_, r) in examples.items()},
        "contrasts": contrasts,
        "largest_loss_state_group": group,
        "interactions": {"n_seconds": int(m), "seed": RNG_SEED, "seconds_per_second_s": round(t_inter / max(m, 1), 4),
                         "row_sum_max_error": row_sum_err, "pairs_reported": len(pairs),
                         "convention": "xgboost off-diagonal entry; the standard total pair effect is twice this"},
        "wall_seconds": round(time.time() - t0, 1),
    }
    json.dump(out_meta, open(os.path.join(OUT, "local_explain_meta.json"), "w"), indent=1)
    print(json.dumps({k: out_meta[k] for k in ["base_probability", "examples", "wall_seconds"]}, indent=1))
    print("interactions:", out_meta["interactions"])


if __name__ == "__main__":
    main()
