#!/usr/bin/env bash
# Runs ON the space, detached: recompute all 47 bucket chunks with the patched code, then compare them with pos_parts.
set -u
cd "$HOME/adam/builder" || exit 1
PY="$HOME/venv-dsdk/bin/python"
export POSB_OUT="$HOME/adam/builder/final/pos_check_all"
mkdir -p "$POSB_OUT"
CH=$(for i in $(seq 0 46); do printf 'c%03d ' "$i"; done)
echo "$(date '+%F %T') recompute start: $CH"
"$PY" pos_bucket.py $CH
echo "$(date '+%F %T') recompute exit=$?"
"$PY" identity_space.py $CH
echo "$(date '+%F %T') identity exit=$?"
