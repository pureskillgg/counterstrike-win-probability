# Run queue (2026-11-02)

Frozen benchmark: 3,059 test matches in scratchpad/frozen_test_keys.json. Never change it. Frozen cache: data/combined/cache_snapshot_2026.parquet (never overwritten).

## In progress
1. Tome build, October 28 (chunk 3 of 4 building), then 29, 30, 31. Log: logs/extend_2026-10-26_to_31.log
2. Per-second layers for October 23-27 (status, then positions). Log: logs/extract_status_oct26_27.log, logs/extract_positions_oct26_27.log
3. Pipeline runner, waiting on the tome markers. Log: logs/pipeline_oct28_31.log

## Pipeline (runner: scratchpad/pipeline/pipeline_oct28_31.sh)
- append 28-31 to the combined store
- backup matches, rounds, grid as *_pre_oct28
- prepare.py, then frozen_check.py (stops the pipeline if any frozen test round changed)
- make_grid.py
- extract 1 Hz status and positions for 28-31
- new cache: data/combined/cache_snapshot_2026_oct23_31.parquet
- retrain.py: control on the frozen cache (must reproduce 0.45643), then on the expanded cache, both scored on the frozen 3,059 test matches

## Next experiments, in order
E1. Money on hand per side, saving per-row predictions, then stratified paired gains by seconds into round, bomb phase, and pre-plant alive configuration. Money was -0.00023 overall; this shows where it lands.
E2. Richer-information floor: per-player status and position features (health, armor, money, equipment, positions) on the same frozen test rows, trained on a fixed subsample of train for both models. This estimates the objectively reducible loss.
E3. Equipment value vs money, to separate the two economy signals (equipment is already in v4).
E4. Positions gain stratified by seconds into round (predictions not saved in the earlier run).
E5. Expanded-train vs frozen-cache paired comparison (from the pipeline's retrain step).

## Dropped
- Loss streak / economy history: the user ruled it out as lossy.
