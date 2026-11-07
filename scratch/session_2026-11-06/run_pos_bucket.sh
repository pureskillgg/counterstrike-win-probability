#!/usr/bin/env bash
# Run ON the space. Processes bucket chunks one after another, detached from the SSH session.
#
#   Start (from the laptop):
#     ssh adam-gpu 'mkdir -p ~/adam/builder/logs && setsid nohup ~/adam/builder/run_pos_bucket.sh c000 c001 c002 >> ~/adam/builder/logs/pos_bucket.log 2>&1 < /dev/null &'
#   Watch:
#     ssh adam-gpu 'tail -n 20 ~/adam/builder/logs/pos_bucket.log'
#
# Resumable: pos_bucket.py skips a chunk whose output file already exists, so after a stop just start again with the same list.
# Survives an SSH disconnect (setsid + nohup). It cannot survive the SageMaker app being shut down.
set -u
if [ "$#" -eq 0 ]; then
  echo "usage: run_pos_bucket.sh c000 [c001 ...]" >&2
  exit 2
fi
cd "$HOME/adam/builder" || exit 1
PY="$HOME/venv-dsdk/bin/python"

for c in "$@"; do
  echo "$(date '+%F %T') start $c"
  if "$PY" pos_bucket.py "$c"; then
    echo "$(date '+%F %T') done $c"
  else
    echo "$(date '+%F %T') FAILED $c (rerun the list later; finished chunks are skipped)"
  fi
done
echo "$(date '+%F %T') all requested chunks processed"
