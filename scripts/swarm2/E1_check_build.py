"""E1: report xgboost build flags and run a tiny CPU and CUDA smoke test.

Run with the C: env interpreter:
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\E1_check_build.py
"""
import platform
import sys

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import sklearn
import xgboost as xgb

print("python", sys.version.split()[0], sys.executable)
print("platform", platform.platform())
print("xgboost", xgb.__version__, xgb.__file__)
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("polars", pl.__version__)
print("pyarrow", pa.__version__)
print("sklearn", sklearn.__version__)

bi = xgb.build_info()
print("build_info USE_CUDA =", bi.get("USE_CUDA"))
print("build_info USE_NCCL =", bi.get("USE_NCCL"))
print("build_info USE_OPENMP =", bi.get("USE_OPENMP"))
print("build_info keys =", sorted(bi.keys()))

X = np.random.default_rng(0).random((2000, 5), dtype=np.float32)
y = (X[:, 0] + X[:, 1] > 1).astype(np.float32)
d = xgb.DMatrix(X, label=y)
for dev in ("cpu", "cuda"):
    try:
        b = xgb.train(
            {"tree_method": "hist", "device": dev, "max_depth": 4, "objective": "binary:logistic"},
            d,
            num_boost_round=5,
        )
        print("device", dev, "OK, boosted rounds =", b.num_boosted_rounds())
    except Exception as e:  # noqa: BLE001 - report any failure verbatim
        print("device", dev, "FAILED:", type(e).__name__, str(e)[:800])
