#!/bin/bash
# GDT scoring of the 47 GDT-trained runs, on a pod. Reads from the volume (read-only), writes to /workspace/results/gdt_scores.
set -e
W=/workspace/gdtscore
B=/root/gdtscore
PY=/workspace/envs/cs2/.pixi/envs/torch/bin/python
mkdir -p $B/root/data/combined $B/out
cp -r $W/root/data/combined/. $B/root/data/combined/
cp $W/build_validation_gdt_rows.py $W/score_gdt.py $W/table.csv $W/frozen_test_keys.json $W/stageB_pooled.py $B/
sed -i 's|ROOT = "G:/pureskill/csds-cs2"|ROOT = "/root/gdtscore/root"|' $B/build_validation_gdt_rows.py
sed -i "s|^SCORE = .*|SCORE = \"$B/score_gdt.py\"|; s|^TABLE = .*|TABLE = \"$B/table.csv\"|; s|^ROWS = .*|ROWS = \"$B/out/rows_gdt.parquet\"|" $B/stageB_pooled.py
ARGS=()
for f in /workspace/results/pos24_gdt_*/*_gdt_seed*_final_val_preds.parquet; do
  b=$(basename "$f"); arm=${b%%_gdt_seed*}; s=$(echo "$b" | sed -E 's/.*_gdt_seed([0-9]+)_final.*/\1/')
  ARGS+=(--model "${arm}_s${s}=$f")
done
ARGS+=(--model "xgb83_s0=/workspace/gdtscore/xgb83_s0_final_val_preds.parquet")
echo "arm-seed prediction files: ${#ARGS[@]} / 2"
cd $B
$PY -I -X utf8 build_validation_gdt_rows.py --cache /workspace/cs2data/combined/cache_snapshot_2026_ext.parquet --frozen $B/frozen_test_keys.json --out $B/out/rows_gdt.parquet "${ARGS[@]}"
$PY -I -X utf8 score_gdt.py --rows $B/out/rows_gdt.parquet --table $B/table.csv --out $B/out/stageA_seed0 --models gru_s0,xlstm_s0,xlstm_fast_s0,xlstm_blk_s0,xlstm_sig_s0,rwkv7_s0,mingru_s0,tf_s0,gdn_s0,minlstm_s0,tcn_s0,mamba2_s0,lru_s0,xlstm_mh_s0,gdn2_s0,lstm_s0,griffin_s0,gru2_s0,tf2_s0,xgb83_s0
$PY -I -X utf8 stageB_pooled.py > $B/out/stageB_pooled_results.txt 2>&1
mkdir -p /workspace/results/gdt_scores && cp -r $B/out/. /workspace/results/gdt_scores/
echo SCORE_DONE
