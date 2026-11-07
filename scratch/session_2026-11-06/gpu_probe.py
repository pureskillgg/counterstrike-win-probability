import sys, time
import numpy as np
import xgboost as xgb
print(sys.version.split()[0], "xgboost", xgb.__version__)
rng = np.random.default_rng(0)
X = rng.random((200000, 20), dtype=np.float32)
y = (X[:, 0] + rng.normal(0, 0.5, 200000) > 0.5).astype(np.int64)
d = xgb.QuantileDMatrix(X, y, max_bin=256)
t = time.time()
b = xgb.train({"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.03, "max_depth": 4, "min_child_weight": 20,
               "subsample": 0.8, "tree_method": "hist", "device": "cuda", "seed": 0}, d, num_boost_round=20)
print("cuda train ok in", round(time.time() - t, 2), "s; trees", b.num_boosted_rounds())
