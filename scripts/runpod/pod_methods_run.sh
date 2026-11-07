#!/usr/bin/env bash
set -e
cd /workspace
if [ ! -d csds-cs2 ]; then tar -xf pod_bundle_20261016_005813.tar; fi
cd /workspace/csds-cs2
mkdir -p logs models
python3 -m venv /workspace/venv --system-site-packages
. /workspace/venv/bin/activate
python -m pip install -q "polars>=1.0" pyarrow "xgboost-cu12==3.2.0" "scikit-learn==1.7.2" scipy
python -X utf8 -c "import xgboost as xgb; print('xgboost', xgb.__version__, 'CUDA', xgb.build_info().get('USE_CUDA'))"
nohup python -X utf8 scripts/model/exp_methods.py --store combined --device cuda --trials 16 --stride-search 2 > logs/exp_methods.log 2>&1 < /dev/null &
echo started