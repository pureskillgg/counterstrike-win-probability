#!/bin/bash
# Run one or more pinned sequence arms on a pod: bash run_arm.sh <arms> <seeds> <tag>
# Results (metrics json and validation predictions) go to the network volume, so they survive the pod.
ARMS=$1; SEEDS=$2; TAG=$3
PY=/workspace/envs/cs2/.pixi/envs/torch/bin/python
OUT=/workspace/results/pos24_${TAG}
mkdir -p $OUT
cd /root/ws
$PY -I -X utf8 -u scripts/model/train_phase1_pos24.py --arms $ARMS --seeds $SEEDS --steps 30000 --eval_every 1000 \
  --dist --no_test_load --out_dir $OUT > $OUT/run.log 2>&1
echo RUN_DONE_$TAG >> $OUT/run.log
