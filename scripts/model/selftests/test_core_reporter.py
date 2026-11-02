"""Selftest of the reporting machinery (checks/core.py): a pass on an empty population is a FAIL, NOT RUN counts as FAIL unless allowed, WARN never fails,
an exception inside a check is a recorded FAIL with its traceback, parameters reject typos, and reports are replaced atomically."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not put the script folder on sys.path
from harness import T  # noqa: E402

from checks import core  # noqa: E402
from checks.core import fail, notrun, ok, skip, warn  # noqa: E402

t = T("reporter, parameters, atomic report")
r = core.Reporter(allow_missing=False, echo=False)
a = r.add("X1", "pass on nothing", ok("looked at nothing", n=0))
t.check("a PASS on a population of size 0 becomes a FAIL (vacuous)", a["status"] == "FAIL" and "VACUOUS" in a["detail"], a["detail"])
a = r.add("X2", "pass on enough", ok("fine", n=10), min_n=5)
t.check("a PASS on a population at the floor stays a PASS", a["status"] == "PASS" and a["passed"])
a = r.add("X3", "below a stated floor", ok("fine", n=3), min_n=5)
t.check("the floor is a parameter", a["status"] == "FAIL")
a = r.add("X4", "missing input", notrun("no file"))
t.check("NOT RUN counts as FAIL by default", a["status"] == "NOT RUN" and not a["passed"])
a = r.add("X5", "known limitation", warn("31% incomplete", n=100))
t.check("WARN passes", a["status"] == "WARN" and a["passed"])
a = r.run("X6", "raises", lambda: 1 // 0)
t.check("an exception becomes a FAIL with the traceback recorded", a["status"] == "FAIL" and "ZeroDivisionError" in a.get("traceback", ""), a["detail"])
t.check("the run is not all_passed and lists the failures", not r.all_passed() and {x["id"] for x in r.failed()} == {"X1", "X3", "X4", "X6"}, str([x["id"] for x in r.failed()]))
r2 = core.Reporter(allow_missing=True, echo=False)
r2.add("Y1", "needs a file", notrun("no file"))
r2.add("Y2", "fast", skip("--fast"))
r2.add("Y3", "limitation", warn("known", n=1))
t.check("with allow_missing: NOT RUN, SKIP and WARN do not fail the run", r2.all_passed(), str(r2.counts()))
t.check("an empty reporter never counts as passed", not core.Reporter().all_passed())
d = tempfile.mkdtemp(prefix="wp_core_")
path = os.path.join(d, "rep.json")
json.dump({"all_passed": True}, open(path, "w"))
core.dump_json_atomic({"all_passed": False, "x": [1, 2]}, path)
t.check("a report replaces the old file atomically and leaves no temp file", json.load(open(path))["all_passed"] is False and os.listdir(d) == ["rep.json"], str(os.listdir(d)))
P = core.Params({"a": 1, "b": 0.5, "c": True, "d": "x"}, ["a=3", "b=0.25", "c=false", "d=y"])
t.check("Params parse ints, floats, booleans and text", (P.a, P.b, P.c, P.d) == (3, 0.25, False, "y"), str(P.as_dict()))
try:
    core.Params({"a": 1}, ["aa=3"])
    t.check("an unknown parameter name is an error (a typo must not keep the default)", False)
except KeyError:
    t.check("an unknown parameter name is an error (a typo must not keep the default)", True)
t.check("peak memory is measurable", (core.peak_rss_mb() or 0) > 1, str(core.peak_rss_mb()))
t.finish()
