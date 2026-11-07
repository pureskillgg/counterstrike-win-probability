#!/usr/bin/env bash
# Laptop side. Pings adam-gpu every few minutes so the space session stays active while you work.
#
#   bash keepalive_adam.sh          # every 240 s (default)
#   bash keepalive_adam.sh 120      # every 120 s
#   Ctrl+C to stop.
#
# Limits: this only sends a short command over SSH. It cannot start a stopped space, and it may not stop
# SageMaker's idle shutdown if the idle timer ignores SSH traffic. It stops when this window closes.
INTERVAL="${1:-240}"
WRAP="/c/Users/13144/Documents/pureskill/counterstrike-win-probability/pipeline/sagemaker/connect_adam_gpu.sh"

if [ ! -x "$WRAP" ] && [ ! -f "$WRAP" ]; then
  echo "wrapper not found: $WRAP" >&2
  exit 1
fi

while true; do
  out=$(bash "$WRAP" 'echo "up $(uptime -p 2>/dev/null || uptime)"' 2>&1)
  rc=$?
  echo "$(date '+%F %T') rc=$rc $(printf '%s' "$out" | tail -n 1)"
  sleep "$INTERVAL"
done
