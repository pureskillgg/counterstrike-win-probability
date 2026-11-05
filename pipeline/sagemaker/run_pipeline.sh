#!/usr/bin/env bash
# Run the whole pipeline in the foreground, in order: chunk build, holdout rebuild, duplicate check.
# Start it from a JupyterLab terminal (bash ~/pipe/run_pipeline.sh) so the terminal session stays active.
# Chunks already in S3 are skipped, so a rerun resumes. The holdout and duplicate check start over.
set -u

PY="$HOME/venv-dsdk/bin/python"
LOG="$HOME/pipe/pipeline_run.log"

{
  echo "== start $(date '+%F %T')"
  "$PY" "$HOME/pipe/chunk_build.py" || { echo "PIPELINE FAILED at chunk build $(date '+%F %T')"; exit 1; }
  "$PY" "$HOME/pipe/holdout_build.py" || { echo "PIPELINE FAILED at holdout build $(date '+%F %T')"; exit 1; }
  "$PY" "$HOME/dup_check/dup_check.py" || { echo "PIPELINE FAILED at duplicate check $(date '+%F %T')"; exit 1; }
  echo "PIPELINE DONE $(date '+%F %T')"
} 2>&1 | tee "$LOG"
