#!/usr/bin/env bash
# Pod side, full-dataset model comparison: install the pinned packages into /workspace/venv and print the versions.
# The RunPod image runs Python 3.11, so xgboost 3.4 and scikit-learn 1.9 (Python 3.12 only) cannot install here: xgboost 3.2.0 is the version
# the v3 export used on the pod, and scikit-learn 1.7.2 is the newest that supports Python 3.11.
# Run:  bash /workspace/csds-cs2/scripts/runpod/pod_compare_setup.sh
set -euo pipefail
cd /workspace/csds-cs2
python3 -m venv /workspace/venv --system-site-packages
# shellcheck disable=SC1091
source /workspace/venv/bin/activate
python -m pip install -q "polars>=1.0" pyarrow "xgboost-cu12==3.2.0" "scikit-learn==1.7.2" "lightgbm==4.7.0" "catboost==1.2.10" scipy
python -c "import catboost, lightgbm, xgboost, sklearn, polars; print('catboost', catboost.__version__, '| lightgbm', lightgbm.__version__, '| xgboost', xgboost.__version__, '| sklearn', sklearn.__version__, '| polars', polars.__version__)"
nvidia-smi --query-gpu=name --format=csv,noheader

