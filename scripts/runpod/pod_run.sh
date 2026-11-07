#!/usr/bin/env bash
# Pod side: set up Python, check the GPU, run the heavy steps of the CS2 win-probability pipeline, pack the outputs.
#   cd /workspace/csds-cs2 && nohup bash pod_run.sh > logs/pod_run.log 2>&1 &     (detached: closing ssh does not stop it)
#   bash pod_run.sh --only export_basic,export_snapshot     (run some steps)      bash pod_run.sh --from baselines
# Progress: logs/chain_state.txt (one line per finished step).  Outputs: /workspace/pod_outputs_<stamp>.tar.gz  (also written after every step).
set -uo pipefail
ROOT="${POD_ROOT:-/workspace/csds-cs2}"
cd "$ROOT" || exit 1
mkdir -p logs models
STATE="logs/chain_state.txt"
ONLY=""; FROM=""
while [ $# -gt 0 ]; do case "$1" in --only) ONLY="$2"; shift 2;; --from) FROM="$2"; shift 2;; *) shift;; esac; done
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATE"; }

if [ ! -d /workspace/venv ]; then
  say "creating venv (system site packages kept: the image already has CUDA libraries)"
  python3 -m venv /workspace/venv --system-site-packages || { say "venv failed"; exit 1; }
fi
# shellcheck disable=SC1091
source /workspace/venv/bin/activate
python -m pip install -q --upgrade pip >/dev/null 2>&1
python -m pip install -q "polars>=1.0" pyarrow numpy "xgboost>=2.0" scikit-learn scipy || { say "pip install failed"; exit 1; }
say "python $(python -V 2>&1); xgboost $(python -c 'import xgboost;print(xgboost.__version__)'); polars $(python -c 'import polars;print(polars.__version__)')"
say "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1); host RAM: $(free -g | awk '/Mem:/{print $2" GB"}'); cores: $(nproc)"
python - <<'PY' || { echo "GPU smoke test FAILED"; exit 1; }
import numpy as np, xgboost as xgb
X = np.random.rand(20000, 8).astype("float32"); y = (X[:, 0] + 0.1 * np.random.rand(20000) > 0.55).astype(int)
m = xgb.XGBClassifier(n_estimators=20, device="cuda", tree_method="hist").fit(X, y)
print("GPU smoke test ok; train acc", float((m.predict(X) == y).mean()))
PY
export OMP_NUM_THREADS="$(nproc)" MKL_THREADING_LAYER=SEQUENTIAL
PY="python -X utf8"
VER="${VER:-winprob_v3}"          # model name for every output of this chain (VER=winprob_v4 bash pod_run.sh ...)
CUTOFF="${CUTOFF:-2026/10/11}"     # temporal experiment: train through this upload day, test on the newer matches

declare -A CMD
CMD[export_basic]="$PY scripts/model/export_model.py --store combined --sets map --name ${VER}_basic --stride 2"
CMD[export_snapshot]="$PY scripts/model/export_model.py --store combined --from-cache --sets map,status --eras 2026 --name ${VER} --stride 2"
CMD[stress]="$PY scripts/model/stress_test_winprob.py --model models/${VER}.json --fallback models/${VER}_basic.json --write-golden --report logs/stress_${VER}.json"
CMD[stress_basic]="$PY scripts/model/stress_test_winprob.py --model models/${VER}_basic.json --write-golden --report logs/stress_${VER}_basic.json"
CMD[inspect]="$PY scripts/model/inspect_rounds.py --pred data/combined/export_${VER}_test_predictions.parquet --store combined --out data/combined/round_inspection_${VER}"
CMD[baselines]="$PY scripts/model/run_baselines.py --store combined --seeds 0,1"
CMD[robustness]="$PY scripts/model/exp_robustness.py --store combined --stride 3"
CMD[temporal]="$PY scripts/model/exp_temporal.py --store combined --cutoff $CUTOFF"
CMD[families]="$PY scripts/model/exp_model_families.py --store combined --stride 6"
CMD[deep]="$PY scripts/model/exp_deep_distill.py --store combined --stride 2 --folds 3 --epochs 4 --width 1024 --depth 4 --dropout 0.3 --lr 0.0005 --wd 0.01 --evals-per-epoch 6 --noise 0.1 --alphas 0.5,0.75 --tag regB"
ORDER=(export_basic export_snapshot stress stress_basic inspect baselines robustness temporal deep families)

pack() {
  local f="/workspace/pod_outputs_$(date -u +%Y%m%d_%H%M%S).tar.gz"
  tar czf "$f" models logs data/combined/export_* data/combined/baselines.json data/combined/exp_*.json data/combined/round_inspection_${VER} 2>/dev/null
  say "packed outputs -> $f ($(du -h "$f" | cut -f1))"
}

started=0; [ -z "$FROM" ] && started=1
for step in "${ORDER[@]}"; do
  [ "$started" = 0 ] && [ "$step" = "$FROM" ] && started=1
  [ "$started" = 0 ] && continue
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$step,"* ]]; then continue; fi
  for try in 1 2; do
    say "start $step (try $try): ${CMD[$step]}"
    t0=$(date +%s)
    # a step that would leave the pod without RAM is killed instead of taking the whole pod down
    ( eval "${CMD[$step]}" > "logs/$step.log" 2>&1 ) &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
      if [ "${avail:-99}" -lt 2 ]; then say "RAM below 2 GB free: killing $step"; pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; break; fi
      sleep 10
    done
    wait "$pid"; code=$?
    say "end $step exit $code ($(( $(date +%s) - t0 )) s)"
    [ "$code" = 0 ] && break
  done
  pack
done
say "chain finished"
