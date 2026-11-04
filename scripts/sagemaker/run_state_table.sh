#!/usr/bin/env bash
# run_state_table.sh
#
# One script, three stages:
#   1. Copy the chunk tomes for the channels the build needs from S3 to NVMe scratch, in parallel.
#   2. Build the event-only state table for each chunk, in parallel.
#   3. Join the chunk tables and write one file under adam/ in S3.
#
# Run it in the live JupyterLab terminal, in the foreground, with the tab kept open.
# Do not background it (nohup, tmux, screen): detached jobs do not keep the space up.
#
#   bash run_state_table.sh 2>&1 | tee /mnt/sagemaker-nvme/state_run/run.log
#
# Resumable: chunks already copied or built are skipped, so after a failure just run it again.
# Scratch is wiped when the space stops, so run it in one session.
set -uo pipefail

# ---------------- CONFIG: fill from the probe before running ----------------
BUCKET="s3://pureskillgg-csgo-production-sagemaker-0"
CHUNK_FROM=0
CHUNK_TO=67
SCRATCH="/mnt/sagemaker-nvme/state_run"
PY="$HOME/venv-dsdk/bin/python"   # environment with pyarrow, used for the final join
COPY_PAR=8                        # parallel copies
BUILD_PAR=4                       # parallel builds; check free -h before raising it

# Raw basic channels the build reads, as in basic_<channel>.<dates>. Example: CHANNELS=(round_end player_death)
# molotov_state is never copied raw. The harmonized file is copied instead, because the raw types differ across chunks.
CHANNELS=()

# Build command, run once per chunk. CHUNK_DIR (local chunk folder) and OUT_FILE (parquet to write) are exported.
# Example: BUILD_CMD='python ~/pipe/scripts/model/build_states.py --chunk "$CHUNK_DIR" --out "$OUT_FILE"'
BUILD_CMD=""

FINAL_S3="$BUCKET/adam/tomes/states/event_only_state.parquet"
# ----------------------------------------------------------------------------

[ "${#CHANNELS[@]}" -gt 0 ] || { echo "CHANNELS is empty. Fill it in from the probe first."; exit 1; }
[ -n "$BUILD_CMD" ] || { echo "BUILD_CMD is empty. Fill it in from the probe first."; exit 1; }

mkdir -p "$SCRATCH/chunks" "$SCRATCH/states" "$SCRATCH/logs"
log() { echo "[$(date +%H:%M:%S)] $*"; }
T0=$(date +%s)

log "host: $(nproc) cores; $(free -h | awk '/Mem:/{print $2" total, "$7" available"}')"
df -h "$SCRATCH" | tail -n 1
aws sts get-caller-identity >/dev/null 2>&1 || { echo "AWS credentials are not working."; exit 1; }

read -r -p "Have you told the admin this job will run over an hour? Type yes to continue: " ans
[ "$ans" = "yes" ] || { echo "Stopped. Tell the admin first."; exit 1; }

INCLUDES=""
for ch in "${CHANNELS[@]}"; do
  [ "$ch" = "molotov_state" ] && continue
  INCLUDES="$INCLUDES --include basic_${ch}.*"
done
export BUCKET SCRATCH PY INCLUDES BUILD_CMD

copy_chunk() {
  c=$(printf "c%03d" "$1")
  dst="$SCRATCH/chunks/$c"
  mkdir -p "$dst"
  for try in 1 2 3; do
    if aws s3 sync "$BUCKET/adam/tomes/chunks/$c/" "$dst/" --exclude "*" --include "DONE.json" $INCLUDES --quiet \
       && aws s3 cp "$BUCKET/adam/tomes/harmonized/molotov_state/$c.parquet" "$dst/harmonized_molotov_state.parquet" --quiet; then
      echo "$c COPIED"
      return 0
    fi
    sleep 5
  done
  echo "$c COPY_FAILED"
  return 1
}
export -f copy_chunk

build_chunk() {
  c=$(printf "c%03d" "$1")
  export CHUNK_DIR="$SCRATCH/chunks/$c"
  export OUT_FILE="$SCRATCH/states/$c.parquet"
  if [ -s "$OUT_FILE" ]; then
    echo "$c SKIPPED (already built)"
    return 0
  fi
  if eval "$BUILD_CMD" > "$SCRATCH/logs/build_$c.log" 2>&1; then
    echo "$c BUILT"
  else
    echo "$c BUILD_FAILED (see $SCRATCH/logs/build_$c.log)"
    return 1
  fi
}
export -f build_chunk

log "stage 1 of 3: copy chunks $CHUNK_FROM to $CHUNK_TO to scratch, $COPY_PAR at a time"
seq "$CHUNK_FROM" "$CHUNK_TO" | xargs -P "$COPY_PAR" -I{} bash -c 'copy_chunk {}' | tee "$SCRATCH/logs/copy.log"

log "stage 2 of 3: build the state table per chunk, $BUILD_PAR at a time"
seq "$CHUNK_FROM" "$CHUNK_TO" | xargs -P "$BUILD_PAR" -I{} bash -c 'build_chunk {}' | tee "$SCRATCH/logs/build.log"

built=$(ls "$SCRATCH/states"/c*.parquet 2>/dev/null | wc -l)
expected=$((CHUNK_TO - CHUNK_FROM + 1))
log "stage 3 of 3: $built of $expected chunks built"
if [ "$built" -ne "$expected" ]; then
  echo "Not every chunk built. Check logs/copy.log and logs/build.log, then run this script again."
  exit 1
fi

"$PY" - <<'PY'
import glob, os
import pyarrow as pa
import pyarrow.parquet as pq
scratch = os.environ["SCRATCH"]
files = sorted(glob.glob(os.path.join(scratch, "states", "c*.parquet")))
tables = [pq.read_table(f) for f in files]
out = pa.concat_tables(tables, promote_options="permissive")
pq.write_table(out, os.path.join(scratch, "event_only_state.parquet"), compression="zstd")
print("chunks joined:", len(files), "rows:", out.num_rows)
PY

aws s3 cp "$SCRATCH/event_only_state.parquet" "$FINAL_S3" && log "written: $FINAL_S3"
aws s3 sync "$SCRATCH/logs" "$BUCKET/adam/tomes/states/logs/" --quiet
echo "copy failures: $(grep -c FAILED "$SCRATCH/logs/copy.log" 2>/dev/null || echo 0)"
echo "build failures: $(grep -c FAILED "$SCRATCH/logs/build.log" 2>/dev/null || echo 0)"
log "total time: $(( $(date +%s) - T0 )) seconds"
