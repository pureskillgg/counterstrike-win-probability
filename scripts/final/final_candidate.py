"""Validation-only comparison for choosing one final candidate (owner ruling: choose on validation, then tell him).

Candidates, all on the same 1,515,228 validation seconds:
- XGBoost full set, depth 4, eta 0.03, min_child_weight 20 (the sprint's selected setting), seeds 0 to 2:
  run/gpu_fast predictions.
- Sequence arms at their final step (fixed 30,000 steps, no checkpoint selection), seeds 0 to 2:
  megarun_*_w*_final_val_preds.parquet and phase1_large_dist (width 128).

Paired per second, arm minus XGBoost candidate, paired() from scripts/model/spatial_feature_eval.py, 1,000 match resamples.
"""
import glob, json, os, sys
import numpy as np, polars as pl

sys.path.insert(0, "G:/pureskill/csds-cs2/scripts/model")
from spatial_feature_eval import paired  # noqa: E402

DIAG = "G:/pureskill/csds-cs2/data/diagnostics"
RUN = "G:/pureskill/csds-cs2/data/sprint/run/gpu_fast"
KEY = ["match_key", "round", "second"]
XGB_ID = "full_d4_e0.03_m20_s{}"
SEQ = {
    "gru_w128 (phase1_large_dist)": (f"{DIAG}/phase1_large_dist", "gru", [0, 1, 2]),
    "gru_w256 (megarun)": (f"{DIAG}/megarun_gru_w256", "gru", [0, 1, 2]),
    "gru_w512 (megarun)": (f"{DIAG}/megarun_gru_w512", "gru", [0, 1, 2]),
    "tcn_w128 (phase1_large_dist)": (f"{DIAG}/phase1_large_dist", "tcn", [0, 1, 2]),
    "tcn_w256 (megarun)": (f"{DIAG}/megarun_tcn_w256", "tcn", [0, 1, 2]),
    "tcn_w512 (megarun)": (f"{DIAG}/megarun_tcn_w512", "tcn", [0, 1, 2]),
    "tf_w128 L4 (594b4126 arm D)": (f"{DIAG}/phase1_large_dist", "tf", [0, 1, 2]),
    "tf_w256 L4 (666e8dca arm G)": (f"{DIAG}/phase1_tf_G_w256_lr3e4_dist", "tf", [0, 1, 2]),
}

def xgb_preds(s):
    return pl.read_parquet(f"{RUN}/preds/{XGB_ID.format(s)}.parquet").select(KEY + ["label_ct_win", "p"]).rename({"p": "px"})

def seq_preds(folder, arm, s):
    return pl.read_parquet(f"{folder}/{arm}_seed{s}_final_val_preds.parquet").select(KEY + ["label", "p"]).rename({"label": "label_ct_win", "p": "ps"})

out = {}
for name, (folder, arm, seeds) in SEQ.items():
    rows, seq_ll = [], []
    for s in seeds:
        if not os.path.exists(f"{folder}/{arm}_seed{s}_final_val_preds.parquet"):
            continue
        x = xgb_preds(s)
        q = seq_preds(folder, arm, s)
        j = q.join(x, on=KEY, how="inner")
        assert j.height == x.height == q.height, f"{name} seed {s}: row sets differ"
        assert (j["label_ct_win"] == j["label_ct_win_right"]).all(), f"{name} seed {s}: labels disagree"
        r = paired(j["match_key"].to_numpy(), j["label_ct_win"].to_numpy().astype(np.float64),
                   j["ps"].to_numpy().astype(np.float64), j["px"].to_numpy().astype(np.float64), seed=0)
        rows.append({"seed": s, "diff": r["diff"], "ci95": r["ci95"]})
        seq_ll.append(float(-(j["label_ct_win"].to_numpy() * np.log(np.clip(j["ps"].to_numpy(), 1e-15, 1))
                              + (1 - j["label_ct_win"].to_numpy()) * np.log(np.clip(1 - j["ps"].to_numpy(), 1e-15, 1))).mean()))
    if rows:
        d = np.array([r["diff"] for r in rows])
        out[name] = {"seeds": [r["seed"] for r in rows], "per_seed": rows, "pooled": float(d.mean()),
                     "seq_ll_mean": float(np.mean(seq_ll)),
                     "all_upper_below_0": all(r["ci95"][1] < 0 for r in rows),
                     "all_lower_above_0": all(r["ci95"][0] > 0 for r in rows)}

xgb_ll = []
for s in [0, 1, 2]:
    x = xgb_preds(s)
    xgb_ll.append(float(-(x["label_ct_win"].to_numpy() * np.log(np.clip(x["px"].to_numpy(), 1e-15, 1))
                          + (1 - x["label_ct_win"].to_numpy()) * np.log(np.clip(1 - x["px"].to_numpy(), 1e-15, 1))).mean()))
out["_xgb_candidate_ll_mean_seeds_0_2"] = float(np.mean(xgb_ll))
json.dump(out, open("C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/final_candidate.json", "w"), indent=1)
for k, v in out.items():
    if k.startswith("_"):
        print(k, round(v, 6))
    else:
        verdict = "XGB better" if v["pooled"] > 0 and v["all_lower_above_0"] else ("sequence better" if v["pooled"] < 0 and v["all_upper_below_0"] else "no claim")
        print(f"{k}: seq LL {v['seq_ll_mean']:.6f}; (seq minus XGB) pooled {v['pooled']:+.6f}; seeds {v['seeds']}; {verdict}")
