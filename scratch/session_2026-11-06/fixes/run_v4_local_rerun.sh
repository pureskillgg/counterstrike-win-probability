#!/bin/bash
# v4 rerun of the local 1 Hz stage (file list narrowed to the local target dates), then assemble and verify.
PY=$HOME/venv-dsdk/bin/python
V4S=/home/sagemaker-user/adam/builder/final/export_clean_v4/scripts
L=/home/sagemaker-user/adam/builder/logs
cd $V4S || { echo "CD FAILED"; exit 1; }
echo "[$(date +%T)] local rerun start (python: $PY)"
$PY local_job_v4.py 2> $L/v4_local.log
rc=$?
echo "[$(date +%T)] local rerun rc=$rc"
if [ "$rc" -ne 0 ]; then echo "LOCAL RERUN FAILED"; exit 1; fi
$PY assemble_v4.py > $L/v4_assemble.log 2>&1
rc=$?
echo "[$(date +%T)] assemble rc=$rc"
if [ "$rc" -ne 0 ]; then echo "ASSEMBLE FAILED"; exit 1; fi
$PY verify_v4.py > $L/v4_verify.log 2>&1
rc=$?
echo "[$(date +%T)] verify rc=$rc"
echo "[$(date +%T)] RERUN DONE"
