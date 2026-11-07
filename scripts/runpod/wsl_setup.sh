#!/usr/bin/env bash
set -e
rm -rf ~/csds_venv
python3 -m venv --without-pip ~/csds_venv
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/csds_venv/bin/python /tmp/get-pip.py -q
. ~/csds_venv/bin/activate
python -m pip install -q "polars>=1.0" pyarrow "xgboost-cu12==3.2.0" "scikit-learn==1.7.2" scipy numpy
cat > /tmp/gpu_check.py <<'PY'
import xgboost as xgb, numpy as np
print("xgboost", xgb.__version__, "CUDA build", xgb.build_info().get("USE_CUDA"))
X = np.random.rand(20000, 8).astype(np.float32); y = (X[:, 0] > 0.5).astype(int)
xgb.XGBClassifier(n_estimators=10, device="cuda", tree_method="hist").fit(X, y)
print("GPU fit ok")
PY
python /tmp/gpu_check.py