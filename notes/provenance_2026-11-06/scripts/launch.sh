#!/bin/bash
# usage: launch.sh <pod_no> <ip> <port>
N=$1; IP=$2; PORT=$3
SSHO="ssh -i $HOME/.ssh/runpod_pureskill -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new -p $PORT root@$IP"
SCP="scp -i $HOME/.ssh/runpod_pureskill -o StrictHostKeyChecking=accept-new -P $PORT"
DIR="$(cd "$(dirname "$0")" && pwd)"
$SCP "$DIR/run_gdt.sh" "$DIR/driver_p$N.sh" root@$IP:/workspace/pod_runner/ && \
$SSHO "chmod +x /workspace/pod_runner/*.sh && mkdir -p /workspace/results && nohup bash -c 'bash /workspace/pod_runner/setup_pod.sh && bash /workspace/pod_runner/driver_p$N.sh' > /workspace/results/driver_p$N.log 2>&1 < /dev/null &" && echo "LAUNCHED pod $N $IP:$PORT"
