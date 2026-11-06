#!/usr/bin/env bash
# Run one command on adam-gpu, e.g.:  pipeline/sagemaker/connect_adam_gpu.sh 'uptime'
#
# The space gets a new SSH host key every time it restarts, so the stored key stops matching.
# When that exact error happens for the adam-gpu alias, this script removes only the adam-gpu
# entry for the space's ARN (backing up known_hosts first), logs the new fingerprint, and retries
# once. It never touches other hosts and never retries more than once.
#
# This trades SSH's host-key check for convenience on this one host. The connection still goes
# through the SageMaker proxy, which authenticates with AWS, so the risk is limited to this alias.
set -u

HOST_ALIAS=adam-gpu
ARN="arn:aws:sagemaker:us-east-1:<account-id>:space/d-<domain-id>/adam-gpu"
KNOWN_HOSTS="${HOME}/.ssh/known_hosts"
LOG="${HOME}/.ssh/adam-gpu-hostkey.log"

run_ssh() {
  ssh -o BatchMode=yes -o ConnectTimeout=60 "$HOST_ALIAS" "$@"
}

out=$(run_ssh "$@" 2>&1)
rc=$?

if [ "$rc" -eq 255 ] && printf '%s' "$out" | grep -q "REMOTE HOST IDENTIFICATION HAS CHANGED"; then
  new_fp=$(printf '%s' "$out" | grep -o 'SHA256:[^ ]*' | head -n 1)
  stamp=$(date '+%Y-%m-%d %H:%M:%S')
  cp "$KNOWN_HOSTS" "${KNOWN_HOSTS}.bak-$(date +%Y%m%d-%H%M%S)"
  ssh-keygen -R "$ARN" >/dev/null 2>&1
  echo "$stamp replaced host key for $ARN; new fingerprint $new_fp" >> "$LOG"
  echo "adam-gpu restarted with a new host key; replaced the stored entry (logged to $LOG)" >&2
  out=$(run_ssh "$@" 2>&1)
  rc=$?
fi

printf '%s\n' "$out"
exit "$rc"
