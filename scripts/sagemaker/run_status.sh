#!/usr/bin/env bash
# Runs the reduced player_status build for all chunks. Resumable: finished chunks are skipped.
# Output is appended to v4_all.log so a rerun keeps the earlier lines.
cd ~/adam/status_scripts || exit 1
~/venv-dsdk/bin/python -W ignore add_player_status_v4.py --chunks 0-67 2>&1 | tee -a ~/adam/status_scripts/v4_all.log
