"""Run every scripts/model/selftests/test_*.py, one process each (isolated: a crashing test cannot hide another). Exit 1 if any fails.

  python -I -X utf8 scripts\\model\\selftests\\run_all.py
"""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
tests = sorted(f for f in os.listdir(here) if f.startswith("test_") and f.endswith(".py"))
if not tests:
    print("FAIL  no selftests found (an empty suite proves nothing)")
    sys.exit(1)
env = dict(os.environ, MKL_THREADING_LAYER="SEQUENTIAL", PYTHONIOENCODING="utf-8")
bad = []
for t in tests:
    r = subprocess.run([sys.executable, "-I", "-X", "utf8", os.path.join(here, t)], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = [x for x in (r.stdout + r.stderr).strip().splitlines() if x.strip()][-1:] or [""]
    print(f"{'PASS' if r.returncode == 0 else 'FAIL'}  {t}  {tail[0][:150]}", flush=True)
    if r.returncode != 0:
        bad.append(t)
print("RESULT:", "ALL PASS" if not bad else f"FAILED {bad}")
sys.exit(1 if bad else 0)
