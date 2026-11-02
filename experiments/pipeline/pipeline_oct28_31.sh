#!/bin/bash
# Queue: add October 28-31 to the combined store, rebuild rounds/grid with a frozen-test guard,
# extract per-second layers, and build a NEW cache file (the frozen cache_snapshot_2026.parquet is never written).
cd /g/pureskill/csds-cs2
PY=.pixi/envs/default/python.exe
SCR=/c/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad
LOG=logs/pipeline_oct28_31.log
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
wait_for() { until [ -f "$1" ]; do sleep 60; done; }
wait_for_text() { until grep -q "$2" "$1" 2>/dev/null; do sleep 60; done; }

log "queue start"
for d in 28 29 30 31; do wait_for "tomes/2026-10-$d.CHUNKS_DONE"; log "tomes 2026-10-$d done"; done
SRC=$(for d in 28 29 30 31; do ls tomes | grep -E "^2026-10-${d}_c[0-9]+$"; done | paste -sd, -)
log "append sources: $SRC"
$PY -I -X utf8 scripts/data/append_combined.py --sources "$SRC" > logs/append_oct28_31.log 2>&1 || { log "append FAILED"; exit 1; }
log "append done"

for f in matches rounds grid; do cp -n data/combined/$f.parquet data/combined/${f}_pre_oct28.parquet; done
$PY -I -X utf8 scripts/data/prepare.py --store combined > logs/prepare_oct28_31.log 2>&1 || { log "prepare FAILED"; exit 1; }
log "prepare done"
$PY -I -X utf8 "C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/pipeline/frozen_check.py" data/combined/rounds_pre_oct28.parquet data/combined/rounds.parquet > logs/frozen_check_oct28_31.log 2>&1
if [ $? -ne 0 ]; then log "FROZEN TEST CHANGED - stopping before grid/cache"; cat logs/frozen_check_oct28_31.log >> "$LOG"; exit 1; fi
log "frozen test unchanged after prepare"

$PY -I -X utf8 scripts/data/make_grid.py --store combined > logs/make_grid_oct28_31.log 2>&1 || { log "make_grid FAILED"; exit 1; }
log "grid done"

wait_for_text logs/extract_positions_oct26_27.log POSITIONS_DONE
log "earlier 1Hz extraction (23-27) finished"
$PY -I -X utf8 scripts/data/extract_status_1hz.py --store combined > logs/extract_status_oct28_31.log 2>&1 || { log "status extract FAILED"; exit 1; }
log "status 1Hz done"
$PY -I -X utf8 scripts/data/extract_positions_1hz.py --store combined > logs/extract_positions_oct28_31.log 2>&1 || { log "positions extract FAILED"; exit 1; }
log "positions 1Hz done"

$PY -I -X utf8 scripts/model/make_snapshot_cache.py --store combined --out-name cache_snapshot_2026_oct23_31.parquet > logs/cache_oct23_31.log 2>&1 || { log "cache build FAILED"; exit 1; }
log "new cache built: data/combined/cache_snapshot_2026_oct23_31.parquet"

# retrain on the frozen split: control on the frozen cache, then the expanded cache
WSLS=/mnt/c/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad
wsl -e bash -lc "cd /mnt/g/pureskill/csds-cs2 && TAG=control_frozen ~/venvs/csds/bin/python -I -X utf8 $WSLS/pipeline/retrain.py > $WSLS/pipeline/control_frozen.log 2>&1"
wsl -e bash -lc "cd /mnt/g/pureskill/csds-cs2 && CACHE=/mnt/g/pureskill/csds-cs2/data/combined/cache_snapshot_2026_oct23_31.parquet TAG=with_oct23_31 ~/venvs/csds/bin/python -I -X utf8 $WSLS/pipeline/retrain.py > $WSLS/pipeline/with_oct23_31.log 2>&1"
log "retrain done; see scratchpad/pipeline/*_result.json"
log "queue end"
