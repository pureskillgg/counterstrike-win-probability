#!/bin/bash
# usage: run_reg.sh <seed> <A|B>. A: 30k steps, dropout 0.2, weight decay 1e-4. B: 25k steps, no regularisation. Log-loss training.
S=$1; V=$2
PY=/workspace/envs/cs2/.pixi/envs/torch/bin/python
if [ "$V" = A ]; then EXTRA="--steps 30000 --dropout 0.2 --weight_decay 1e-4"; else EXTRA="--steps 25000"; fi
OUT=/workspace/results/reg_${V}
mkdir -p $OUT
cd /root/ws
$PY -I -X utf8 -u scripts/model/train_phase1_pos24.py --arms gru --seeds $S $EXTRA --eval_every 1000 --dist --no_test_load --out_dir $OUT >> $OUT/run.log 2>&1
echo RUN_DONE_${V}_${S} >> $OUT/run.log
