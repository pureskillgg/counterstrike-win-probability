#!/bin/bash
# v4 chain: copy the v3 arrays into v4, S3 channel build (c000-c067), local 1 Hz build, assemble, verify.
PY=$HOME/venv-dsdk/bin/python
V3=/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout
V4=/home/sagemaker-user/adam/builder/final/export_clean_v4
V4S=$V4/scripts
L=/home/sagemaker-user/adam/builder/logs
echo "[$(date +%T)] chain start pid $$"
cp $V3/loadout_step_i16.npy $V3/loadout_equip_i32.npy $V3/match_keys.json $V3/offsets.npy $V4/ || { echo "COPY FAILED"; exit 1; }
cp $V3/scripts/*.py $V4S/ || { echo "SCRIPT COPY FAILED"; exit 1; }
echo "[$(date +%T)] copy of v3 arrays into v4 done"
cd $V4S || exit 1
$PY s3_job_v4.py > $V4S/s3_cov_v4.csv 2> $L/v4_s3_channel.log; rc=$?
echo "[$(date +%T)] S3 channel build rc=$rc"
[ $rc -eq 0 ] || { echo "S3 BUILD FAILED"; exit $rc; }
$PY local_job_v4.py 2> $L/v4_local.log; rc=$?
echo "[$(date +%T)] local 1 Hz build rc=$rc"
[ $rc -eq 0 ] || { echo "LOCAL BUILD FAILED"; exit $rc; }
$PY assemble_v4.py > $L/v4_assemble.log 2>&1; rc=$?
echo "[$(date +%T)] assemble rc=$rc"
[ $rc -eq 0 ] || { echo "ASSEMBLE FAILED"; exit $rc; }
$PY verify_v4.py > $L/v4_verify.log 2>&1; rc=$?
echo "[$(date +%T)] verify rc=$rc"
echo "[$(date +%T)] CHAIN DONE"
