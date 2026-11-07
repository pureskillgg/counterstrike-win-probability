#!/bin/bash
# Waits for the seed-0 architecture screen to finish everywhere it runs, then exits so the caller is notified.
#   laptop: mingru, minlstm, lru (metrics json in pos24_screen_new)
#   pod-03: xlstm, rwkv7 (RUN_DONE_screen_xr in run.log)
#   pod-04: griffin, mamba2 (RUN_DONE_screen_gm in run.log)
# Any Traceback is reported on every poll so a crashed arm is seen, not silently waited on.
LAPTOP_DIR="G:/pureskill/csds-cs2/data/diagnostics/pos24_screen_new"
SSHO=(ssh -i "$HOME/.ssh/runpod_pureskill" -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new)
MAX_POLLS=40
for i in $(seq 1 $MAX_POLLS); do
  L=$(ls "$LAPTOP_DIR"/*_metrics.json 2>/dev/null | wc -l)
  P3=$("${SSHO[@]}" -p <port> root@<pod-ip> 'grep -c RUN_DONE_screen_xr /workspace/results/pos24_screen_xr/run.log 2>/dev/null || echo 0' 2>/dev/null | tail -n 1)
  P4=$("${SSHO[@]}" -p <port> root@<pod-ip> 'grep -c RUN_DONE_screen_gm /workspace/results/pos24_screen_gm/run.log 2>/dev/null || echo 0' 2>/dev/null | tail -n 1)
  T3=$("${SSHO[@]}" -p <port> root@<pod-ip> 'grep -c Traceback /workspace/results/pos24_screen_xr/run.log 2>/dev/null || echo 0' 2>/dev/null | tail -n 1)
  T4=$("${SSHO[@]}" -p <port> root@<pod-ip> 'grep -c Traceback /workspace/results/pos24_screen_gm/run.log 2>/dev/null || echo 0' 2>/dev/null | tail -n 1)
  echo "$(date '+%T') poll=$i laptop_metrics=$L pod03_done=${P3:-0} pod04_done=${P4:-0} pod03_tracebacks=${T3:-0} pod04_tracebacks=${T4:-0}"
  if [ "${L:-0}" -ge 3 ] && [ "${P3:-0}" -ge 1 ] && [ "${P4:-0}" -ge 1 ]; then
    echo SCREEN_DONE
    exit 0
  fi
  sleep 240
done
echo SCREEN_TIMEOUT
exit 1
