#!/usr/bin/env bash
# Checkpointed chain for a pod: (1) positions by round type, (2) every-n-seconds sampling. Restart-safe: a finished model or cell
# loads its saved predictions from data/combined/ckpt/ instead of training again. Run it again after a restart; nothing is lost
# except the model that was training when the pod stopped.
# The launcher extracts the bundle into /workspace before this runs, so this script never overwrites itself while it runs.
# Checkpoints load only when their settings fingerprint matches (see wp_common.fingerprint); otherwise the model is trained again.
# cache_pos_2026.parquet is moved into place here if the launcher uploaded it to /workspace.
cd /workspace/csds-cs2
mkdir -p logs data/combined/ckpt
if [ -f /workspace/cache_pos_2026.parquet ]; then mv -f /workspace/cache_pos_2026.parquet data/combined/; fi
if [ ! -d /workspace/venv ]; then python3 -m venv /workspace/venv --system-site-packages; fi
. /workspace/venv/bin/activate
python -m pip install -q "polars>=1.0" pyarrow "xgboost-cu12==3.2.0" "scikit-learn==1.7.2" scipy
python -X utf8 -c "import xgboost as xgb; print('xgboost', xgb.__version__, 'CUDA build', xgb.build_info().get('USE_CUDA'))"
echo "start $(date)" >> logs/status.txt
python -X utf8 scripts/model/exp_positions_groups.py --store combined --device cuda >> logs/positions_groups.log 2>&1
echo "positions_groups exit $? $(date)" >> logs/status.txt
python -X utf8 scripts/model/exp_tick_sampling.py --store combined --device cuda --seeds 2 >> logs/tick_sampling.log 2>&1
echo "tick_sampling exit $? $(date)" >> logs/status.txt
echo "chain finished $(date)" >> logs/status.txt
