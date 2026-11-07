#!/bin/bash
# Run GDT-objective arms on a pod (pin 64d46c76): bash run_gdt.sh <arms> <seeds> <tag>
# Results go to the network volume: /workspace/results/pos24_<tag>. A RUN_DONE_<tag> marker is appended when the run ends.
ARMS=$1; SEEDS=$2; TAG=$3
PY=/workspace/envs/cs2/.pixi/envs/torch/bin/python
OUT=/workspace/results/pos24_${TAG}
mkdir -p $OUT
cd /root/ws
$PY -I -X utf8 -u scripts/model/train_phase1_pos24.py --arms $ARMS --seeds $SEEDS --steps 30000 --eval_every 1000 \
  --dist --no_test_load --objective gdt --out_dir $OUT >> $OUT/run.log 2>&1
echo RUN_DONE_$TAG >> $OUT/run.log
