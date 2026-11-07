"""verify E1-2 (env probe): report xgboost build flags and run a 2-tree CUDA smoke test.

Run with the interpreter named on the command line; prints which build it is.
Writes nothing to disk.
"""
import sys

import numpy as np
import xgboost as xgb

print("exe", sys.executable, flush=True)
print("python", sys.version.split()[0], "numpy", np.__version__, flush=True)
print("xgboost", xgb.__version__, flush=True)
bi = xgb.build_info()
print("USE_CUDA", bi.get("USE_CUDA"), "CUDA_VERSION", bi.get("CUDA_VERSION"),
      "USE_OPENMP", bi.get("USE_OPENMP"), flush=True)

rng = np.random.default_rng(0)
X = rng.standard_normal((2000, 5), dtype=np.float32)
y = (X[:, 0] > 0).astype(np.float32)
for dev in ("cpu", "cuda"):
    try:
        b = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": dev,
                       "max_depth": 2, "nthread": 1},
                      xgb.DMatrix(X, label=y), num_boost_round=2)
        cfg = b.save_config()
        print(f"smoke device={dev} OK config_device_found={'\"device\":\"cuda' in cfg or 'cuda' in cfg}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"smoke device={dev} FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
