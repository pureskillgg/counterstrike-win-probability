#!/bin/bash
# Upload export_clean_v4 to the RunPod network volume z7yvc40ykx (EU-RO-1), key prefix export_clean_v4/.
# Excludes steps_alive.npy (undocumented copy of v2, disagrees with loadouts; not for training).
# Writes a local sha256 manifest, then the verify step re-downloads everything and compares.
set -u
SRC=/home/sagemaker-user/adam/builder/final/export_clean_v4
UP=/home/sagemaker-user/adam/builder/fixes/readiness/s3_upload.py
OUTD=/home/sagemaker-user/adam/builder/fixes/readiness/upload_v4
mkdir -p "$OUTD"
cd "$SRC" || exit 1
find . -type f ! -name steps_alive.npy | sort > "$OUTD/file_list.txt"
echo "files: $(wc -l < "$OUTD/file_list.txt")  bytes: $(cat $(sed 's|^\./||' "$OUTD/file_list.txt") | wc -c)"
: > "$OUTD/upload.log"
while read -r f; do
  rel=${f#./}
  MAXC=16 ~/venv-dsdk/bin/python "$UP" "$rel" "export_clean_v4/$rel" >> "$OUTD/upload.log" 2>&1 || echo "FAILED $rel" >> "$OUTD/upload.log"
done < "$OUTD/file_list.txt"
echo UPLOAD_DONE >> "$OUTD/upload.log"
