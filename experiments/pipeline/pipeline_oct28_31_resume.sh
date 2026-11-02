#!/bin/bash
# Resume the Oct 28-31 queue after the prepare.py side-cast fix. Append (03:43) and prepare (rerun) are done;
# frozen_check passed (3059 matches, 0 changed). Continues: make_grid, 1 Hz status and positions for 28-31,
# new cache (never the frozen one), then retrain control on the frozen cache and the expanded cache.
cd /g/pureskill/csds-cs2
PY=.pixi/envs/default/python.exe
SCR=/c/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad
LOG=logs/pipeline_oct28_31.log
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }

log "resume: prepare fixed (side cast to Int64), frozen_check passed; continuing from make_grid"
$PY -I -X utf8 scripts/data/make_grid.py --store combined > logs/make_grid_oct28_31.log 2>&1 || { log "make_grid FAILED"; exit 1; }
log "grid done"
$PY -I -X utf8 scripts/data/extract_status_1hz.py --store combined > logs/extract_status_oct28_31.log 2>&1 || { log "status extract FAILED"; exit 1; }
log "status 1Hz done"
$PY -I -X utf8 scripts/data/extract_positions_1hz.py --store combined > logs/extract_positions_oct28_31.log 2>&1 || { log "positions extract FAILED"; exit 1; }
log "positions 1Hz done"
$PY -I -X utf8 scripts/model/make_snapshot_cache.py --store combined --out-name cache_snapshot_2026_oct23_31.parquet > logs/cache_oct23_31.log 2>&1 || { log "cache build FAILED"; exit 1; }
log "new cache built: data/combined/cache_snapshot_2026_oct23_31.parquet"

WSLS=/mnt/c/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad
wsl -e bash -lc "cd /mnt/g/pureskill/csds-cs2 && TAG=control_frozen ~/venvs/csds/bin/python -I -X utf8 $WSLS/pipeline/retrain.py > $WSLS/pipeline/control_frozen.log 2>&1"
wsl -e bash -lc "cd /mnt/g/pureskill/csds-cs2 && CACHE=/mnt/g/pureskill/csds-cs2/data/combined/cache_snapshot_2026_oct23_31.parquet TAG=with_oct23_31 ~/venvs/csds/bin/python -I -X utf8 $WSLS/pipeline/retrain.py > $WSLS/pipeline/with_oct23_31.log 2>&1"
log "retrain done; see scratchpad/pipeline/*_result.json"
log "queue end"
