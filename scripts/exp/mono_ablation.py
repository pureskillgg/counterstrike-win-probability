"""Which monotone constraint costs log loss? Baseline XGBoost settings on the per-second cache, GPU, four constraint sets."""
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
print("features:", list(tr["feat"]), flush=True)
cfgs = {
    "mono_all": [1, -1, 1, 0, 0, 0, 0, 1, 0, 0],
    "mono_none": None,
    "mono_alive_only": [1, -1, 1, 0, 0, 0, 0, 0, 0, 0],
    "mono_score_only": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
}
base = dict(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
            tree_method="hist", device="cuda", early_stopping_rounds=100, eval_metric="logloss", random_state=0)
os.makedirs(r"G:\pureskill\csds-cs2\data\viz\pt_preds", exist_ok=True)
for name, mono in cfgs.items():
    params = dict(base)
    if mono is not None:
        params["monotone_constraints"] = tuple(mono)
    t0 = time.time()
    for attempt in range(3):
        try:
            m = xgb.XGBClassifier(**params)
            m.fit(tr["X"], tr["y"], eval_set=[(va["X"], va["y"])], verbose=False)
            break
        except Exception as exc:
            print("retry", attempt, str(exc)[:160], flush=True)
            time.sleep(30)
    it = (0, int(m.best_iteration) + 1)
    pv = m.predict_proba(va["X"], iteration_range=it)[:, 1]
    pt = m.predict_proba(te["X"], iteration_range=it)[:, 1]
    np.save(rf"G:\pureskill\csds-cs2\data\viz\pt_preds\MONO_{name}_val.npy", pv)
    np.save(rf"G:\pureskill\csds-cs2\data\viz\pt_preds\MONO_{name}_test.npy", pt)
    E.report("MONO_" + name, te, pt, extra={"best_iter": int(m.best_iteration), "fit_s": round(time.time() - t0, 1)})
print("done", flush=True)
