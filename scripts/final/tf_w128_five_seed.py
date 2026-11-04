"""Pre-registration 179e83e4: transformer width 128 (594b4126 arm D), five seeds, against the XGBoost candidate.

Validation seconds only. Same inputs and join as scratchpad/final_candidate.py (88d9fbb0).

Pinned matched comparison, seeds 0 to 2: arm minus candidate, paired per second, paired() with 1,000 match
resamples, RNG seed 0. Positive = candidate better.
Rule (7db520a7): candidate better if pooled >= +0.0005 with every seed's lower bound > 0; transformer better if
pooled <= -0.0005 with every upper bound < 0; otherwise no claim.

Seeds 3 and 4: the sprint ran candidate seeds 0 to 2 only, so no matched candidate exists for these seeds. Each is
compared with the per-second mean of candidate seeds 0 to 2 (probabilities averaged). Reported separately; not the
pinned matched comparison. The averaged comparator is stronger than one candidate seed, which favours the candidate.
"""
import json, sys
import numpy as np, polars as pl

sys.path.insert(0, "G:/pureskill/csds-cs2/scripts/model")
from spatial_feature_eval import paired  # noqa: E402

DIAG = "G:/pureskill/csds-cs2/data/diagnostics/phase1_large_dist"
RUN = "G:/pureskill/csds-cs2/data/sprint/run/gpu_fast"
KEY = ["match_key", "round", "second"]
OUT = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/tf_w128_five_seed.json"


def xgb(s):
    return pl.read_parquet(f"{RUN}/preds/full_d4_e0.03_m20_s{s}.parquet").select(KEY + ["label_ct_win", "p"]).rename({"p": "px"})


def seq(s):
    return pl.read_parquet(f"{DIAG}/tf_seed{s}_final_val_preds.parquet").select(KEY + ["label", "p"]).rename({"label": "label_ct_win", "p": "ps"})


def logloss(y, p):
    p = np.clip(p, 1e-15, 1.0)
    q = np.clip(1.0 - p, 1e-15, 1.0)
    return float(-(y * np.log(p) + (1 - y) * np.log(q)).mean())


def verdict(rows):
    pooled = float(np.mean([r["diff"] for r in rows]))
    all_lower = all(r["ci95"][0] > 0 for r in rows)
    all_upper = all(r["ci95"][1] < 0 for r in rows)
    if pooled >= 0.0005 and all_lower:
        v = "candidate better"
    elif pooled <= -0.0005 and all_upper:
        v = "transformer better"
    else:
        v = "no claim"
    return {"pooled": pooled, "verdict": v}


out = {"matched_seeds_0_2": [], "descriptive_seeds_3_4": [], "candidate_ll_seed": {}, "transformer_ll_seed": {}}

# Pinned matched comparison, seeds 0 to 2.
for s in (0, 1, 2):
    x = xgb(s)
    q = seq(s)
    j = q.join(x, on=KEY, how="inner")
    assert j.height == x.height == q.height, f"seed {s}: row sets differ"
    assert (j["label_ct_win"] == j["label_ct_win_right"]).all(), f"seed {s}: labels disagree"
    y = j["label_ct_win"].to_numpy().astype(np.float64)
    ps = j["ps"].to_numpy().astype(np.float64)
    px = j["px"].to_numpy().astype(np.float64)
    r = paired(j["match_key"].to_numpy(), y, ps, px, seed=0)
    out["matched_seeds_0_2"].append({"seed": s, "diff": float(r["diff"]), "ci95": [float(v) for v in r["ci95"]]})
    out["candidate_ll_seed"][s] = logloss(y, px)
    out["transformer_ll_seed"][s] = logloss(y, ps)

# Candidate per-second mean over seeds 0 to 2, for seeds 3 and 4 only.
c = [xgb(s).select(KEY + ["px"]).rename({"px": f"px{s}"}) for s in (0, 1, 2)]
cm = c[0].join(c[1], on=KEY, how="inner").join(c[2], on=KEY, how="inner")
assert cm.height == c[0].height == c[1].height == c[2].height, "candidate seeds have different row sets"
cm = cm.with_columns(((pl.col("px0") + pl.col("px1") + pl.col("px2")) / 3.0).alias("px_mean")).select(KEY + ["px_mean"])

for s in (3, 4):
    q = seq(s)
    j = q.join(cm, on=KEY, how="inner")
    assert j.height == cm.height == q.height, f"seed {s}: row sets differ from candidate"
    y = j["label_ct_win"].to_numpy().astype(np.float64)
    ps = j["ps"].to_numpy().astype(np.float64)
    pm = j["px_mean"].to_numpy().astype(np.float64)
    r = paired(j["match_key"].to_numpy(), y, ps, pm, seed=0)
    out["descriptive_seeds_3_4"].append({"seed": s, "diff": float(r["diff"]), "ci95": [float(v) for v in r["ci95"]]})
    out["transformer_ll_seed"][s] = logloss(y, ps)

out["candidate_ll_mean_seeds_0_2"] = float(np.mean(list(out["candidate_ll_seed"].values())))
out["transformer_ll_mean_seeds_0_2"] = float(np.mean([out["transformer_ll_seed"][s] for s in (0, 1, 2)]))
out["transformer_ll_mean_seeds_0_4"] = float(np.mean([out["transformer_ll_seed"][s] for s in (0, 1, 2, 3, 4)]))
out["verdict_pinned_matched_0_2"] = verdict(out["matched_seeds_0_2"])
out["verdict_descriptive_0_4"] = verdict(out["matched_seeds_0_2"] + out["descriptive_seeds_3_4"])

json.dump(out, open(OUT, "w"), indent=1)

print("candidate LL per seed (0-2):", {k: round(v, 6) for k, v in out["candidate_ll_seed"].items()})
print("transformer LL per seed:", {k: round(v, 6) for k, v in out["transformer_ll_seed"].items()})
print(f"candidate LL mean seeds 0-2: {out['candidate_ll_mean_seeds_0_2']:.6f}")
print(f"transformer LL mean seeds 0-2: {out['transformer_ll_mean_seeds_0_2']:.6f}")
print(f"transformer LL mean seeds 0-4: {out['transformer_ll_mean_seeds_0_4']:.6f}")
for r in out["matched_seeds_0_2"]:
    print(f"MATCHED seed {r['seed']}: transformer minus candidate {r['diff']:+.6f} ci95 [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]")
for r in out["descriptive_seeds_3_4"]:
    print(f"DESCRIPTIVE seed {r['seed']} vs candidate mean: transformer minus candidate {r['diff']:+.6f} ci95 [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]")
v = out["verdict_pinned_matched_0_2"]
print(f"PINNED (matched seeds 0-2): pooled {v['pooled']:+.6f}; {v['verdict']}")
v = out["verdict_descriptive_0_4"]
print(f"DESCRIPTIVE (seeds 0-4, seeds 3-4 vs candidate mean): pooled {v['pooled']:+.6f}; {v['verdict']}")
