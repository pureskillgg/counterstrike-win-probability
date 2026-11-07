#!/usr/bin/env bash
# Space-side keepalive. Run it from the space's home directory (~), it writes a timestamp every N seconds.
#
#   Start (detached, survives closing the terminal):
#     cd ~ && nohup bash ~/keepalive_space.sh 240 > /dev/null 2>&1 < /dev/null &
#   See it is alive:
#     tail -n 3 ~/keepalive.log
#   Stop it:
#     kill "$(cat ~/keepalive.pid)"
#
# Limit: it keeps this space's activity log moving. It does not stop SageMaker's own idle shutdown
# unless the idle timer counts background activity like this.
INTERVAL="${1:-240}"
LOG="$HOME/keepalive.log"
echo "$$" > "$HOME/keepalive.pid"
echo "$(date '+%F %T') started, every ${INTERVAL}s" >> "$LOG"
while true; do
  echo "$(date '+%F %T') alive" >> "$LOG"
  sleep "$INTERVAL"
done
