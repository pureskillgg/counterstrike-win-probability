"""S2 step 4: the 2026 snapshot feature table (57-input columns plus keys) for the holdout matches.

Runs make_snapshot_cache.py with --store holdout_b: grid, matches and rounds are the holdout-only copies in data/holdout_b/, and the
loadout/status features read data/holdout_b/status_1hz. Output: data/holdout_b/cache_snapshot_2026_h.parquet.
"""
import os
import runpy
import sys

ROOT = r"G:\pureskill\csds-cs2"
sys.argv = ["make_snapshot_cache.py", "--store", "holdout_b", "--status-dir", os.path.join(ROOT, "data", "holdout_b", "status_1hz"),
            "--out-name", "cache_snapshot_2026_h.parquet"]
runpy.run_path(os.path.join(ROOT, "scripts", "model", "make_snapshot_cache.py"), run_name="__main__")
