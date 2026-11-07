#!/usr/bin/env bash
# Copies export_clean_v2 (about 1.5 GB) from the SageMaker space to a RunPod pod's network volume, then checks every file.
# Runs on the laptop in Git Bash. The data streams space -> laptop -> pod, nothing is kept on the laptop.
#
#   bash copy_to_pod.sh <pod_ip> <pod_ssh_port>
#
# If the space's host key changed, first run:  bash pipeline/sagemaker/connect_adam_gpu.sh 'true'
set -u
POD_IP="${1:?usage: copy_to_pod.sh <pod_ip> <pod_ssh_port>}"
POD_PORT="${2:?usage: copy_to_pod.sh <pod_ip> <pod_ssh_port>}"
SRC_DIR="/home/sagemaker-user/adam/builder/final"
DEST="/workspace/cs2/winprob_v1"
SSHPOD=(ssh -p "$POD_PORT" -o StrictHostKeyChecking=accept-new -o BatchMode=yes "root@$POD_IP")
CHECK="$(mktemp -d)"

echo "$(date '+%F %T') copying export_clean_v2 to $DEST on the pod"
ssh -o BatchMode=yes adam-gpu "cd $SRC_DIR && tar cf - export_clean_v2" \
  | "${SSHPOD[@]}" "mkdir -p $DEST && cd $DEST && tar xf - --no-same-owner"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "copy failed (rc=$rc); nothing was verified" >&2
  exit "$rc"
fi

echo "$(date '+%F %T') verifying checksums on both sides"
ssh -o BatchMode=yes adam-gpu "cd $SRC_DIR/export_clean_v2 && sha256sum *" | sort > "$CHECK/space.sha"
"${SSHPOD[@]}" "cd $DEST/export_clean_v2 && sha256sum *" | sort > "$CHECK/pod.sha"
n_src=$(wc -l < "$CHECK/space.sha")
n_dst=$(wc -l < "$CHECK/pod.sha")
if diff -q "$CHECK/space.sha" "$CHECK/pod.sha" > /dev/null; then
  echo "$(date '+%F %T') OK: $n_dst files, checksums identical on space and pod"
else
  echo "$(date '+%F %T') MISMATCH: space has $n_src files, pod has $n_dst; differences:" >&2
  diff "$CHECK/space.sha" "$CHECK/pod.sha" >&2
  exit 1
fi
rm -rf "$CHECK"
