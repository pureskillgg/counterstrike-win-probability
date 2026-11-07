#!/usr/bin/env bash
# Validation numbers the decisions were judged without: the round-type models (V1, V2, V2b against one model) and the every-n-seconds sweep.
# Both on the pod's GPU. Progress: logs/pod_val_chain.log.
set -uo pipefail
cd /workspace/csds-cs2
mkdir -p logs
python3 -m venv /workspace/venv --system-site-packages
. /workspace/venv/bin/activate
python -m pip install -q "polars>=1.0" pyarrow "xgboost-cu12==3.2.0" "scikit-learn==1.7.2" scipy
echo "start $(date)" > logs/pod_val_chain.log
/workspace/venv/bin/python -X utf8 scripts/model/exp_pistol_val.py --device cuda > logs/pod_pistol_val.log 2>&1
echo "pistol validation exit $? $(date)" >> logs/pod_val_chain.log
/workspace/venv/bin/python -X utf8 scripts/model/exp_tick_sampling.py --device cuda > logs/pod_tick_sampling.log 2>&1
echo "sweep exit $? $(date)" >> logs/pod_val_chain.log
