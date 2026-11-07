"""Re-score the saved 57-input booster on the VALIDATION rows of the sprint matrix only.
The matrix is ordered fit, early stop, validation; rows [n_fit + n_es:] are validation rows (sprint_prepare asserts holdout B and the
frozen test are out). No cache, holdout or test file is opened. Output is a NEW file in the scratch folder, not a pre-existing one."""
import json, os
import numpy as np
import polars as pl
import xgboost as xgb

OUT = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/gdt57v83"
ROOT = "G:/pureskill/csds-cs2"
MODEL = f"{ROOT}/models/winprob_v6_drop_heavy.json"
META57 = f"{ROOT}/models/winprob_v6_drop_heavy.meta.json"
MTX = f"{ROOT}/data/sprint/matrix_trainctr"
VAL = f"{MTX}/val_keys.parquet"

meta57 = json.load(open(META57, encoding="utf-8"))
feats57 = meta57["features"]
assert len(feats57) == 57
best = int(meta57["best_iteration"])
rec_ll = float(meta57["validation_logloss"])

mm = json.load(open(f"{MTX}/meta.json", encoding="utf-8"))
feats_all = mm["feats_all"]
n_fit, n_es, n_val, rows = mm["n_fit"], mm["n_es"], mm["n_val"], mm["rows"]
idx = [feats_all.index(f) for f in feats57]
X = np.load(f"{MTX}/X.npy", mmap_mode="r")
y = np.load(f"{MTX}/y.npy", mmap_mode="r")
assert X.shape == (rows, len(feats_all)) and y.shape[0] == rows
a2 = n_fit + n_es
assert rows - a2 == n_val
Xv = np.ascontiguousarray(X[a2:][:, idx]).astype(np.float32)
yv = np.asarray(y[a2:]).astype(np.int64)

V = pl.read_parquet(VAL)
assert V.height == n_val == Xv.shape[0]
assert np.array_equal(V["label_ct_win"].to_numpy().astype(np.int64), yv), "label alignment failed"

bst = xgb.Booster()
bst.load_model(MODEL)
booster_names = bst.feature_names
dm = xgb.DMatrix(Xv, feature_names=feats57)
p = bst.predict(dm, iteration_range=(0, best + 1)).astype(np.float64)
assert p.shape[0] == n_val and not np.isnan(p).any()

def ll(p, y, eps):
    pc = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())

ll_1e15 = ll(p, yv.astype(np.float64), 1e-15)
ll_1e7 = ll(p, yv.astype(np.float64), 1e-7)
out = pl.DataFrame({"match_key": V["match_key"], "round": V["round"], "second": V["second"],
                    "label_ct_win": V["label_ct_win"], "p": p})
path = os.path.join(OUT, "val_rescored_v6_drop_heavy_57.parquet")
out.write_parquet(path)
res = {"file": path, "source": "saved booster models/winprob_v6_drop_heavy.json re-scored on sprint matrix validation rows (new file)",
       "booster_feature_names_present": booster_names is not None, "best_iteration_used": best, "rows": out.height,
       "matches": out["match_key"].n_unique(), "validation_logloss_eps1e-15": ll_1e15, "validation_logloss_eps1e-7": ll_1e7,
       "validation_logloss_recorded_in_meta": rec_ll, "diff_eps1e-7_minus_recorded": ll_1e7 - rec_ll,
       "diff_eps1e-15_minus_recorded": ll_1e15 - rec_ll, "labels_aligned_with_matrix_y": True,
       "matrix_rows": rows, "n_fit": n_fit, "n_early_stop": n_es, "n_validation": n_val}
json.dump(res, open(os.path.join(OUT, "rescore57_val_check.json"), "w", encoding="utf-8"), indent=2)
print(json.dumps(res, indent=2), flush=True)
