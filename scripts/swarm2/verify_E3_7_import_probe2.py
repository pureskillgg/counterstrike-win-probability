"""verify_E3_7 import probe 2: same module list as verify_E3_7_import_probe.py, but bytecode is
ALLOWED (no sys.dont_write_bytecode). Run only with -X pycache_prefix pointing at the session
scratchpad, so any .pyc written goes there and never into either env copy.

Run by verify_E3_7_import_prefix.py. Prints one JSON line.
"""
import json
import os
import sys
import time

MODS = ["numpy", "pandas", "polars", "xgboost", "sklearn"]

out = {"exe": sys.executable, "prefix": sys.prefix,
       "pycache_prefix": sys.pycache_prefix}
out["MKL_THREADING_LAYER_at_start"] = os.environ.get("MKL_THREADING_LAYER")
t_all = time.perf_counter()
per = {}
for m in MODS:
    t0 = time.perf_counter()
    try:
        __import__(m)
        per[m] = round(time.perf_counter() - t0, 3)
    except Exception as exc:
        per[m] = "ERR %s: %s" % (type(exc).__name__, str(exc)[:120])
out["per_module_s"] = per
out["total_import_s"] = round(time.perf_counter() - t_all, 3)
print(json.dumps(out))
sys.stdout.flush()
