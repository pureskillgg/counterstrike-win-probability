#!/bin/bash
# usage: relaunch2.sh <driver name, e.g. driver_A> <ip> <port>. Copies the driver, then runs setup and the driver fully detached.
N=$1; IP=$2; PORT=$3
D="$(cd "$(dirname "$0")" && pwd)"
timeout 120 scp -i $HOME/.ssh/runpod_pureskill -o StrictHostKeyChecking=accept-new -P $PORT "$D/$N.sh" root@$IP:/workspace/pod_runner/ || { echo "SCP_FAILED $N"; exit 1; }
timeout 120 ssh -i $HOME/.ssh/runpod_pureskill -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new -p $PORT root@$IP "chmod +x /workspace/pod_runner/*.sh; mkdir -p /workspace/results; setsid nohup bash -c 'bash /workspace/pod_runner/setup_pod.sh && bash /workspace/pod_runner/$N.sh' > /workspace/results/$N.log 2>&1 < /dev/null &"; echo "launch exit $? for $N on $IP:$PORT"
