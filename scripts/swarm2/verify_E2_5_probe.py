"""verify_E2_5 probe: print effective CatBoost and LightGBM settings used by the E2 benchmark.

Read-only with respect to the project. Run with the E2 workspace interpreter (the project env has no catboost):
  <cs2gpu_alt>\\envs\\default\\python.exe -B -I -X utf8 scripts\\swarm2\\verify_E2_5_probe.py
Writes nothing to disk (allow_writing_files=False, -B avoids bytecode writes).
"""
import sys

import numpy as np
import catboost as cb
import lightgbm as lgb

print("python", sys.executable)
print("catboost", cb.__version__, "| lightgbm", lgb.__version__, "| numpy", np.__version__)

rng = np.random.default_rng(20241)
X = rng.standard_normal((2000, 50), dtype=np.float32)
y = (rng.random(2000) < 0.5).astype(np.int32)

m = cb.CatBoostClassifier(iterations=3, depth=6, learning_rate=0.1, loss_function="Logloss",
                          border_count=128, random_seed=0, verbose=0, allow_writing_files=False,
                          task_type="CPU", thread_count=1)
m.fit(X, y)
p = m.get_all_params()
keys = ["boosting_type", "bootstrap_type", "bagging_temperature", "subsample", "sampling_frequency",
        "leaf_estimation_method", "leaf_estimation_iterations", "leaf_estimation_backtracking",
        "l2_leaf_reg", "border_count", "thread_count", "task_type", "grow_policy", "depth",
        "min_data_in_leaf", "max_leaves", "random_strength", "rsm", "score_function",
        "boost_from_average", "has_time", "fold_permutation_block", "model_size_reg"]
for k in keys:
    print("catboost_param", k, "=", p.get(k, "<absent>"))

b = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, max_bin=255, seed=0,
                   verbose=-1, device_type="cpu", num_threads=1),
              lgb.Dataset(X, label=y, free_raw_data=False), num_boost_round=3)
print("lightgbm_num_trees", b.num_trees())
print("lightgbm_params_passed num_threads=1 num_leaves=63 max_bin=255 (defaults: min_data_in_leaf=20, "
      "min_sum_hessian_in_leaf=1e-3, lambda_l2=0, bagging_fraction=1.0, feature_fraction=1.0, max_depth=-1)")
