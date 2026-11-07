"""verify_E3_7 import probe: times imports in ONE interpreter process; prints one JSON line.

Run by verify_E3_7_import.py (do not run by hand unless needed):
  <env>/python.exe -I -X utf8 verify_E3_7_import_probe.py
Bytecode is not written (sys.dont_write_bytecode below). No files are written.
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True

MODS = ["numpy", "pandas", "polars", "xgboost", "sklearn"]

out = {"exe": sys.executable, "prefix": sys.prefix}
out["MKL_THREADING_LAYER_at_start"] = os.environ.get("MKL_THREADING_LAYER")
t_all = time.perf_counter()
per = {}
for m in MODS:
    t0 = time.perf_counter()
    try:
        __import__(m)
        per[m] = round(time.perf_counter() - t0, 3)
    except Exception as exc:  # record, do not hide
        per[m] = "ERR %s: %s" % (type(exc).__name__, str(exc)[:120])
out["per_module_s"] = per
out["total_import_s"] = round(time.perf_counter() - t_all, 3)
out["MKL_THREADING_LAYER_after_imports"] = os.environ.get("MKL_THREADING_LAYER")
try:
    import xgboost
    bi = xgboost.build_info()
    out["xgboost_version"] = xgboost.__version__
    out["xgboost_USE_CUDA"] = bi.get("USE_CUDA")
except Exception as exc:
    out["xgboost_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
print(json.dumps(out))
sys.stdout.flush()
