#!/bin/bash
# usage: run_w.sh "<arms>" "<seeds>" <bin>. GDT-objective retrain that saves final-step weights (--save_weights).
ARMS=$1; SEEDS=$2; BIN=$3
PY=/workspace/envs/cs2/.pixi/envs/torch/bin/python
OUT=/workspace/results/weights_${BIN}
mkdir -p $OUT
cd /root/ws
$PY -I -X utf8 -u scripts/model/train_phase1_pos24.py --arms $ARMS --seeds $SEEDS --steps 30000 --eval_every 1000 \
  --dist --no_test_load --objective gdt --save_weights --out_dir $OUT >> $OUT/run.log 2>&1
echo RUN_DONE_$3_${ARMS} >> $OUT/run.log
