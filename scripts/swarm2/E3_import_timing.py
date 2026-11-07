"""E3 import timing: how long the pipeline stack takes to import from a given env.

Read-only with respect to the env (run with -B so no .pyc files are written).
Each module is imported once per process and timed separately, so the first run in a
fresh process is a cold-ish import (OS page cache may still be warm from earlier runs).

Usage:
  <env>\\python.exe -I -B -X utf8 E3_import_timing.py <label>
Prints one JSON line.
"""
import json
import sys
import time

label = sys.argv[1] if len(sys.argv) > 1 else "unlabelled"
mods = ["numpy", "pandas", "polars", "pyarrow", "duckdb", "sklearn", "xgboost", "lightgbm"]
out = {"label": label, "exe": sys.executable, "prefix": sys.prefix, "times_s": {}}
t_all = time.perf_counter()
for m in mods:
    t0 = time.perf_counter()
    try:
        __import__(m)
        out["times_s"][m] = round(time.perf_counter() - t0, 3)
    except Exception as exc:  # record, do not hide
        out["times_s"][m] = "ERROR: %s: %s" % (type(exc).__name__, str(exc)[:200])
out["total_import_s"] = round(time.perf_counter() - t_all, 3)
try:
    import xgboost
    out["xgboost_version"] = xgboost.__version__
    bi = xgboost.build_info()
    out["xgboost_USE_CUDA"] = bi.get("USE_CUDA")
    out["xgboost_USE_NCCL"] = bi.get("USE_NCCL")
except Exception as exc:
    out["xgboost_build_info_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
print(json.dumps(out))
sys.stdout.flush()
