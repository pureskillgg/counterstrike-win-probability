#!/bin/bash
# usage: pod_wait_chain.sh <pod_id> <arms> <seeds> <tag>
# Waits for the pod's SSH endpoint, then runs setup_pod.sh and the given arm chain detached on the pod.
POD=$1; ARMS=$2; SEEDS=$3; TAG=$4
S="/c/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
for i in $(seq 1 90); do
  read IP PORT < <(python "$S/pod_endpoints.py" "$POD" 2>/dev/null | python -c "
import sys, re
line = sys.stdin.read()
ip = re.search(r'ip= (\S+)', line)
port = re.search(r\"'22': (\d+)\", line)
print(ip.group(1) if ip else '', port.group(1) if port else '')
")
  if [ -n "$IP" ] && [ -n "$PORT" ]; then
    echo "$(date '+%T') endpoint $IP:$PORT"
    ssh -i "$HOME/.ssh/runpod_pureskill" -p "$PORT" -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new root@"$IP" \
      "setsid nohup bash -c 'bash /workspace/pod_runner/setup_pod.sh > /root/setup.out 2>&1 && bash /workspace/pod_runner/run_arm.sh $ARMS $SEEDS $TAG' > /root/chain.out 2>&1 < /dev/null & echo CHAIN_STARTED"
    echo "$(date '+%T') chain started on $POD"
    exit 0
  fi
  sleep 20
done
echo "$(date '+%T') endpoint never appeared for $POD"
exit 1
