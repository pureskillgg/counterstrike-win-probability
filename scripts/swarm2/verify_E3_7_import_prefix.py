"""verify_E3_7 control: bytecode-equalized import timing, G: copy vs C: copy.

Both copies run with the SAME -X pycache_prefix (a scratchpad directory), so each copy's bytecode
is compiled once into that prefix and then reused. This removes the per-copy __pycache__ state
(C: copy lacks polars and xgboost .pyc, see verify_E3_7_pyc.py) as a variable. MKL_THREADING_LAYER
is set to SEQUENTIAL for both copies, so the threading layer is also equal.

Procedure: one warm-up run per copy (writes the prefix), then ORDER repeated twice (12 timed runs).
Writes only to the scratchpad prefix directory. Nothing is written into either env copy.

Run:
  G:/pureskill/csds-cs2/.pixi/envs/default/python.exe -I -X utf8 G:/pureskill/csds-cs2/scripts/swarm2/verify_E3_7_import_prefix.py
"""
import json
import os
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True  # this driver itself writes no bytecode

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "verify_E3_7_import_probe2.py")
PREFIX = ("C:\\Users\\13144\\AppData\\Local\\Temp\\claude\\C--Users-13144-Documents-pureskill"
          "\\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\\scratchpad\\pyc_prefix")
PY = {
    "G": "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe",
    "C": "C:\\Users\\13144\\.pixi\\workspace-envs\\csds-cs2-15611918389529754563\\envs\\default\\python.exe",
}
ORDER = ["G", "C", "C", "G", "G", "C"]
TIMEOUT = 900


def out(s=""):
    print(s, flush=True)


def run(lab):
    env = dict(os.environ)
    env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    t0 = time.perf_counter()
    p = subprocess.run([PY[lab], "-I", "-X", "utf8", "-X", "pycache_prefix=" + PREFIX, PROBE],
                       capture_output=True, text=True, env=env, timeout=TIMEOUT,
                       encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    line = (p.stdout or "").strip().splitlines()
    rec = None
    if line:
        try:
            rec = json.loads(line[-1])
        except Exception:
            rec = None
    return p.returncode, wall, rec, (p.stderr or "").strip()


def main():
    os.makedirs(PREFIX, exist_ok=True)
    out("verify_E3_7_import_prefix.py  interpreter=%s" % sys.executable)
    out("pycache_prefix (scratchpad, both copies) = %s" % PREFIX)
    out("MKL_THREADING_LAYER forced to SEQUENTIAL for both copies")
    out("")
    out("== warm-up (writes prefix once per copy) ==")
    for lab in ["G", "C"]:
        rc, wall, rec, err = run(lab)
        out("  copy=%s rc=%s wall=%.1fs total=%s  %s" % (lab, rc, wall,
            rec.get("total_import_s") if rec else None, ("" if rec else err[-200:])))
    out("")
    out("== timed runs (ABBA order x2, fresh process each) ==")
    res = {"G": [], "C": []}
    for rep in range(2):
        for i, lab in enumerate(ORDER):
            rc, wall, rec, err = run(lab)
            if rec is None:
                out("  rep=%d run=%d copy=%s rc=%s NO JSON: %s" % (rep, i, lab, rc, err[-200:]))
                continue
            res[lab].append(rec)
            out("  rep=%d run=%d copy=%s rc=%s wall=%.1fs total=%.3fs per_module=%s"
                % (rep, i, lab, rc, wall, rec["total_import_s"], json.dumps(rec["per_module_s"])))
    out("")
    out("== summary (median of timed runs) ==")
    for lab in ["G", "C"]:
        totals = [r["total_import_s"] for r in res[lab]]
        if not totals:
            out("  copy=%s no successful runs" % lab)
            continue
        out("  copy=%s n=%d median_total=%.3fs min=%.3fs max=%.3fs" % (
            lab, len(totals), statistics.median(totals), min(totals), max(totals)))
        for m in ["numpy", "pandas", "polars", "xgboost", "sklearn"]:
            vals = [r["per_module_s"].get(m) for r in res[lab]]
            nums = [v for v in vals if isinstance(v, (int, float))]
            if nums:
                out("      %-8s median=%.3fs  n=%d" % (m, statistics.median(nums), len(nums)))
    out("")
    out("done (scratchpad prefix only; no env copy written)")


if __name__ == "__main__":
    main()
