"""Tiny test harness shared by the selftests/test_*.py files (not a test itself). Usage:

    from harness import T
    t = T("what this file tests")
    t.check("name", ok, "detail")
    t.res("name", core.Res)           # a checks.core result
    t.finish()                        # prints RESULT and exits 1 on any failure
Every selftest is run as its own process with  python -I -X utf8 scripts\\model\\selftests\\test_*.py  and must finish in seconds.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
DATA = os.path.join(os.path.dirname(os.path.dirname(MODEL)), "scripts", "data")
for p in (MODEL, DATA):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("POLARS_MAX_THREADS", "2")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


class T:
    def __init__(self, title):
        self.title, self.fails, self.n, self.t0 = title, [], 0, time.time()
        print(f"== {title}", flush=True)

    def check(self, name, ok, detail=""):
        self.n += 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {str(detail)[:300]}", flush=True)
        if not ok:
            self.fails.append(name)

    def res(self, name, res):
        self.check(name, res.status in ("PASS", "WARN"), f"[{res.status}] {res.detail}")

    def expect_fail(self, name, statuses, ids):
        """statuses: {check id: status} of a run on deliberately damaged data; every id in `ids` must be FAIL (a check that stays green on damage is blind)."""
        miss = [i for i in ids if statuses.get(i) != "FAIL"]
        self.check(name, not miss, f"expected FAIL on {ids}; got { {i: statuses.get(i) for i in ids} }")

    def finish(self):
        print(f"RESULT: {'ALL PASS' if not self.fails else 'FAILED ' + str(self.fails)}  ({self.n} checks, {time.time() - self.t0:.1f}s)", flush=True)
        sys.exit(1 if self.fails else 0)
