#!/bin/bash
# usage: relaunch.sh <pod_no> <ip> <port>. Copies that pod's driver, then starts setup and its driver fully detached.
N=$1; IP=$2; PORT=$3
D="$(cd "$(dirname "$0")" && pwd)"
SSHO="timeout 120 ssh -i $HOME/.ssh/runpod_pureskill -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new -p $PORT root@$IP"
timeout 120 scp -i $HOME/.ssh/runpod_pureskill -o StrictHostKeyChecking=accept-new -P $PORT "$D/driver_p$N.sh" root@$IP:/workspace/pod_runner/ || { echo "SCP_FAILED pod $N"; exit 1; }
$SSHO "chmod +x /workspace/pod_runner/*.sh; mkdir -p /workspace/results; setsid nohup bash -c 'bash /workspace/pod_runner/setup_pod.sh && bash /workspace/pod_runner/driver_p$N.sh' > /workspace/results/driver_p$N.log 2>&1 < /dev/null &" ; echo "relaunch exit $? for pod $N"
