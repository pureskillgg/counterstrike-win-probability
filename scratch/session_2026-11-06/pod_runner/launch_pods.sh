#!/bin/bash
# wait for all 7 position tables on the volume, then refresh them on both pods and start the promotion seeds
export AWS_SHARED_CREDENTIALS_FILE="C:/Users/<user>/Downloads/credentials.txt"
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
while true; do
  n=$(aws --profile runpods3 --endpoint-url https://s3api-eu-ro-1.runpod.io --region eu-ro-1 s3 ls s3://z7yvc40ykx/cs2data/diagnostics/allmaps_pos24_trainsites_grid/ 2>/dev/null | wc -l)
  [ "$n" -ge 7 ] && break
  sleep 60
done
SSHK=~/.ssh/runpod_pureskill
for hp in "<pod-ip> <port> tf" "<pod-ip> <port> tcn"; do
  set -- $hp
  ssh -o BatchMode=yes -o ConnectTimeout=30 -i $SSHK -p $2 root@$1 "rm -rf /root/ws/data/diagnostics/allmaps_pos24_trainsites_grid && cp -r /workspace/cs2data/diagnostics/allmaps_pos24_trainsites_grid /root/ws/data/diagnostics/ && ls /root/ws/data/diagnostics/allmaps_pos24_trainsites_grid | wc -l && nohup bash /workspace/pod_runner/run_arm.sh $3 1,2,3,4 ${3}_s1234 > /root/run_$3.out 2>&1 < /dev/null & echo started_$3"
done
echo LAUNCH_DONE
