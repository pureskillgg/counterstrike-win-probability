#!/usr/bin/env bash
set -e
cd /workspace
tar -xf pod_bundle_20261016_005813.tar
cd /workspace/csds-cs2
mkdir -p logs models
python3 -m venv /workspace/venv --system-site-packages
. /workspace/venv/bin/activate
python -m pip install -q "polars>=1.0" pyarrow "xgboost-cu12==3.2.0" "scikit-learn==1.7.2" "lightgbm==4.7.0" "catboost==1.2.10" scipy
python -X utf8 -c "import xgboost as xgb; print('xgboost', xgb.__version__, 'CUDA build', xgb.build_info().get('USE_CUDA'))"
nohup python -X utf8 scripts/model/exp_full_positions.py --store combined --device cuda > logs/full_positions.log 2>&1 < /dev/null &
echo started
