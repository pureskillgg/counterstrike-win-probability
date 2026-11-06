#!/bin/bash
# Set up one RunPod pod for the sequence screen. Data is read from the network volume (/workspace) and unpacked to the
# pod's local disk (/root/ws), which is much faster to read than the network volume. Idempotent.
set -e
WS=/root/ws
mkdir -p $WS/data/round_state $WS/data/diagnostics $WS/scripts $WS/out
if [ ! -f $WS/scripts/model/train_phase1_pos24.py ]; then
  cp -r /workspace/pod_runner/ws_scripts/model $WS/scripts/
  cp /workspace/pod_runner/ws_scripts/model/train_phase1_pos24.py $WS/scripts/model/ 2>/dev/null || true
fi
# round-state arrays: gunzip the staged .gz files from the volume, 8 at a time
if [ $(ls $WS/data/round_state 2>/dev/null | wc -l) -lt 50 ]; then
  cd /workspace/cs2data_gz/data/round_state
  ls *.gz | xargs -P 8 -I{} sh -c 'gunzip -c "{}" > "'$WS'/data/round_state/$(basename "{}" .gz)"'
fi
# combined inputs the loader reads (round labels and the extended snapshot cache); missing these crashed the first pod runs
mkdir -p $WS/data/combined
for f in rounds.parquet cache_snapshot_2026_ext.parquet; do
  if [ ! -f $WS/data/combined/$f ]; then cp /workspace/cs2data/combined/$f $WS/data/combined/; fi
done
# per-map position inputs (24 columns on every second)
if [ ! -d $WS/data/diagnostics/allmaps_pos24_trainsites_grid ]; then
  cp -r /workspace/cs2data/diagnostics/allmaps_pos24_trainsites_grid $WS/data/diagnostics/
fi
echo SETUP_DONE
ls $WS/data/round_state | wc -l
du -sh $WS/data | cut -f1
