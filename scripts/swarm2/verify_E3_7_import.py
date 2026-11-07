"""verify_E3_7 import driver: paired runs of the import probe on the G: and C: env copies.

Conditions:
  as_is  : the environment as this process has it. The G: copy's sitecustomize.py sets
           MKL_THREADING_LAYER=SEQUENTIAL; the C: copy has no sitecustomize and gets nothing.
  forced : MKL_THREADING_LAYER=SEQUENTIAL set explicitly for both copies, which removes the
           sitecustomize difference (the confound).
Order per condition: G, C, C, G, G, C (three ABBA-style pairs). Each run is a fresh process.
Also runs a numpy matmul probe per copy per condition to test the crash claim in sitecustomize.py.

Run:
  G:/pureskill/csds-cs2/.pixi/envs/default/python.exe -I -X utf8 G:/pureskill/csds-cs2/scripts/swarm2/verify_E3_7_import.py
"""
import json
import os
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "verify_E3_7_import_probe.py")
MATMUL = os.path.join(HERE, "verify_E3_7_matmul_probe.py")
PY = {
    "G": "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe",
    "C": "C:\\Users\\<user>\\.pixi\\workspace-envs\\csds-cs2-15611918389529754563\\envs\\default\\python.exe",
}
ORDER = ["G", "C", "C", "G", "G", "C"]
TIMEOUT = 900


def out(s=""):
    print(s, flush=True)


def env_for(cond):
    env = dict(os.environ)
    if cond == "forced":
        env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    else:
        # as_is: do not set anything; G copy gets its own default from sitecustomize.py
        env.pop("MKL_THREADING_LAYER", None)
    return env


def run_probe(exe, script, cond):
    t0 = time.perf_counter()
    try:
        p = subprocess.run([exe, "-I", "-X", "utf8", script], capture_output=True, text=True,
                           env=env_for(cond), timeout=TIMEOUT, encoding="utf-8", errors="replace")
        wall = time.perf_counter() - t0
        return p.returncode, wall, (p.stdout or "").strip(), (p.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return -999, time.perf_counter() - t0, "", "TIMEOUT after %ds" % TIMEOUT


def main():
    out("verify_E3_7_import.py driver  interpreter=%s" % sys.executable)
    out("pre-existing MKL_THREADING_LAYER in driver env: %r" % os.environ.get("MKL_THREADING_LAYER"))
    out("")
    out("== A. Matmul crash check (one fresh process per copy per condition) ==")
    for cond in ["as_is", "forced"]:
        for lab in ["G", "C"]:
            rc, wall, so, se = run_probe(PY[lab], MATMUL, cond)
            tail = so.splitlines()[-1] if so else ("(no stdout) " + se[-200:])
            out("  cond=%-6s copy=%s rc=%s wall=%.1fs  %s" % (cond, lab, rc, wall, tail))
    out("")
    out("== B. Paired import timing (fresh process per run) ==")
    results = {"as_is": {"G": [], "C": []}, "forced": {"G": [], "C": []}}
    for cond in ["as_is", "forced"]:
        for i, lab in enumerate(ORDER):
            rc, wall, so, se = run_probe(PY[lab], PROBE, cond)
            try:
                rec = json.loads(so.splitlines()[-1])
            except Exception:
                out("  cond=%s run=%d copy=%s rc=%s NO JSON: %s" % (cond, i, lab, rc, se[-300:]))
                continue
            results[cond][lab].append(rec)
            out("  cond=%-6s run=%d copy=%s rc=%s wall=%.1fs total_import=%ss MKL_start=%s MKL_after=%s"
                % (cond, i, lab, rc, wall, rec.get("total_import_s"),
                   rec.get("MKL_THREADING_LAYER_at_start"), rec.get("MKL_THREADING_LAYER_after_imports")))
            out("      per_module=%s" % json.dumps(rec.get("per_module_s")))
    out("")
    out("== C. Summary (median of runs; n is the number of runs) ==")
    for cond in ["as_is", "forced"]:
        for lab in ["G", "C"]:
            recs = results[cond][lab]
            if not recs:
                out("  cond=%-6s copy=%s no successful runs" % (cond, lab))
                continue
            totals = [r["total_import_s"] for r in recs]
            out("  cond=%-6s copy=%s n=%d median_total=%.2fs min=%.2fs max=%.2fs all=%s"
                % (cond, lab, len(totals), statistics.median(totals), min(totals), max(totals), totals))
    out("")
    out("done (no files written by this driver; probes run with -I -X utf8 and sys.dont_write_bytecode)")


if __name__ == "__main__":
    main()
