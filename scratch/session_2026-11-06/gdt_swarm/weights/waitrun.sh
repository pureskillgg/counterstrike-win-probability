#!/bin/bash
# usage: waitrun.sh <driver to wait for> <driver to run next>. Waits for the running driver's process, then starts the next one.
PAT=$1; NEXT=$2
PID=$(pgrep -f "^bash /workspace/pod_runner/$PAT.sh" | tail -n1)
if [ -n "$PID" ]; then while kill -0 $PID 2>/dev/null; do sleep 30; done; fi
bash /workspace/pod_runner/$NEXT.sh
