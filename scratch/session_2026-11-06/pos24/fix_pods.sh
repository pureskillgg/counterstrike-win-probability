#!/bin/bash
# Pods failed at start: the sequence loader reads data/combined/rounds.parquet and cache_snapshot_2026_ext.parquet,
# which were never put on the volume. Upload both from the laptop, copy them into each pod's /root/ws, relaunch the arms.
set -u
TMPDIR="${TMPDIR:-/tmp}"; export TMPDIR
AWSOPT="--profile runpod --endpoint-url https://s3api-eu-ro-1.runpod.io --region eu-ro-1"
B=s3://z7yvc40ykx/cs2data/combined
L="G:/pureskill/csds-cs2/data/combined"
RUN_ARM="C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/pod_runner/run_arm.sh"
P1=(ssh -i ~/.ssh/runpod_pureskill -p 44377 -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new root@213.173.102.201)
P2=(ssh -i ~/.ssh/runpod_pureskill -p 17191 -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new root@213.173.110.152)

echo "$(date '+%T') uploading rounds.parquet and cache_snapshot_2026_ext.parquet"
aws $AWSOPT s3 cp "$L/rounds.parquet" "$B/rounds.parquet" > /dev/null 2>"$TMPDIR/up1.err" &
pid1=$!
aws $AWSOPT s3 cp "$L/cache_snapshot_2026_ext.parquet" "$B/cache_snapshot_2026_ext.parquet" > /dev/null 2>"$TMPDIR/up2.err" &
pid2=$!
wait $pid1; rc1=$?
wait $pid2; rc2=$?
echo "$(date '+%T') upload exit codes: rounds=$rc1 cache=$rc2"
if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then
  cat "$TMPDIR/up1.err" "$TMPDIR/up2.err" >&2
  echo "UPLOAD_FAILED"; exit 1
fi

echo "$(date '+%T') volume listing"
aws $AWSOPT s3 ls "$B/" | grep -E "rounds|cache_snapshot_2026_ext"

echo "$(date '+%T') copying into pod workspaces"
for P in P1 P2; do
  eval "CMD=(\"\${$P[@]}\")"
  "${CMD[@]}" 'mkdir -p /root/ws/data/combined && cp /workspace/cs2data/combined/rounds.parquet /workspace/cs2data/combined/cache_snapshot_2026_ext.parquet /root/ws/data/combined/ && ls -la /root/ws/data/combined/rounds.parquet /root/ws/data/combined/cache_snapshot_2026_ext.parquet && ls /root/ws/data/round_state | grep -c _ext.npz'
done

echo "$(date '+%T') launching arms"
"${P1[@]}" 'cat > /root/run_arm.sh && chmod +x /root/run_arm.sh' < "$RUN_ARM"
"${P2[@]}" 'cat > /root/run_arm.sh && chmod +x /root/run_arm.sh' < "$RUN_ARM"
"${P1[@]}" 'setsid nohup bash /root/run_arm.sh tf 1,2,3,4 tf_s1234 > /root/run_tf.out 2>&1 < /dev/null &'
"${P2[@]}" 'setsid nohup bash /root/run_arm.sh tcn 1,2,3,4 tcn_s1234 > /root/run_tcn.out 2>&1 < /dev/null &'

echo "$(date '+%T') waiting 240 s for the first step lines"
sleep 240
echo "== pod01 tf run.log"; "${P1[@]}" 'tail -n 6 /workspace/results/pos24_tf_s1234/run.log'
echo "== pod02 tcn run.log"; "${P2[@]}" 'tail -n 6 /workspace/results/pos24_tcn_s1234/run.log'
echo "FIX_PODS_DONE"
