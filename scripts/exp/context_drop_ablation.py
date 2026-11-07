"""Does the model need match-context features (score lead, round number, pistol, overtime)?

Per-second cache, baseline XGBoost settings, GPU, alive-count monotone constraints only (the unconstrained-score setup
that scored 0.4781 in mono_ablation.py). Variants drop columns by name; each variant is fit with 3 seeds.
Seed-0 predictions go through pt_eval.report (cluster bootstrap, appended to data/viz/pt_results.jsonl as CTX_<name>).
"""
import json
import os
import sys
import time

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")
import numpy as np
import xgboost as xgb

import pt_eval as E

D = E.load()
tr, va, te = D["train"], D["val"], D["test"]
FEAT = [str(f) for f in tr["feat"]]
print("features:", FEAT, flush=True)

ALIVE_MONO = {"alive_ct": 1, "alive_t": -1, "ct_alive_minus_t": 1}
VARIANTS = {
    "all10": [],
    "drop_score": ["ct_score_minus_t_score"],
    "drop_score_round": ["ct_score_minus_t_score", "round_number"],
    "drop_all_context": ["ct_score_minus_t_score", "round_number", "is_pistol_round", "is_overtime"],
}
SEEDS = [0, 1, 2]
base = dict(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
            tree_method="hist", device="cuda", early_stopping_rounds=100, eval_metric="logloss")
os.makedirs(r"G:\pureskill\csds-cs2\data\viz\pt_preds", exist_ok=True)

summary = {}
for name, drop in VARIANTS.items():
    keep = [i for i, f in enumerate(FEAT) if f not in drop]
    kept_names = [FEAT[i] for i in keep]
    mono = tuple(ALIVE_MONO.get(f, 0) for f in kept_names)
    lls = []
    for seed in SEEDS:
        t0 = time.time()
        for attempt in range(3):
            try:
                m = xgb.XGBClassifier(**base, random_state=seed, monotone_constraints=mono)
                m.fit(tr["X"][:, keep], tr["y"], eval_set=[(va["X"][:, keep], va["y"])], verbose=False)
                break
            except Exception as exc:  # GPU is shared with another job; retry on transient CUDA errors
                print("retry", attempt, str(exc)[:160], flush=True)
                time.sleep(30)
        it = (0, int(m.best_iteration) + 1)
        pt = m.predict_proba(te["X"][:, keep], iteration_range=it)[:, 1]
        ll = float(E.ll_vec(pt, te["y"]).mean())
        lls.append(ll)
        print(f"{name} seed {seed}: test logloss {ll:.4f} best_iter {m.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
        if seed == 0:
            np.save(rf"G:\pureskill\csds-cs2\data\viz\pt_preds\CTX_{name}_test.npy", pt)
            E.report("CTX_" + name, te, pt, extra={"kept": kept_names, "best_iter": int(m.best_iteration)})
    summary[name] = {"kept": kept_names, "logloss_by_seed": lls, "mean": float(np.mean(lls)), "range": float(np.ptp(lls))}

print("SUMMARY", json.dumps(summary, indent=1), flush=True)
with open(r"G:\pureskill\csds-cs2\data\viz\context_drop_ablation.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1)
