#!/usr/bin/env bash
# Baseline families on a RunPod pod, run after the local WSL VM kept running out of memory during the data load.
# The pod's RAM floor is 64 GB. Part A: XGBoost (reloads the GPU fit saved as winprob_v4), LightGBM, logistic regression,
# histogram boosting, random forest, ExtraTrees, MLP. Part B: CatBoost alone. Then the merge into data/combined/exp_full_comparison.json.
# Progress: logs/pod_chain.log. Each part writes its own log in logs/.
set -uo pipefail
cd /workspace/csds-cs2
mkdir -p logs
python3 -m venv /workspace/venv --system-site-packages
. /workspace/venv/bin/activate
python -m pip install -q "polars>=1.0" pyarrow "xgboost-cu12==3.2.0" "scikit-learn==1.7.2" "lightgbm==4.7.0" "catboost==1.2.10" scipy
python -c "import catboost, lightgbm, xgboost, sklearn, polars; print('catboost', catboost.__version__, '| lightgbm', lightgbm.__version__, '| xgboost', xgboost.__version__, '| sklearn', sklearn.__version__, '| polars', polars.__version__)" > logs/pod_versions.log 2>&1
echo "start $(date)" > logs/pod_chain.log
python -X utf8 scripts/model/exp_full_comparison.py --models xgb,lgbm,logit,hgb,rf,et,mlp --xgb-model winprob_v4 --out exp_full_comparison_part_a.json > logs/pod_part_a.log 2>&1
echo "part A exit $? $(date)" >> logs/pod_chain.log
python -X utf8 scripts/model/exp_full_comparison.py --models xgb,catboost --xgb-model winprob_v4 --out exp_full_comparison_part_b.json > logs/pod_part_b.log 2>&1
echo "part B exit $? $(date)" >> logs/pod_chain.log
python -X utf8 scripts/model/merge_comparison.py exp_full_comparison_part_a.json exp_full_comparison_part_b.json > logs/pod_merge.log 2>&1
echo "merge exit $? $(date)" >> logs/pod_chain.log
