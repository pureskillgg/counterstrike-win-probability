# -*- coding: utf-8 -*-
"""verify_E3_3_runtime2.py

Follow-up to verify_E3_3_runtime.py: the first run's crash cases printed nothing
(stdout was block-buffered and lost on the native crash), so the stage where the
process dies was unknown. This re-run uses python -u, prints a stage marker
after each step, and adds one test on the claimed C: env with the same
environment variable that the G: sitecustomize.py sets (a candidate fix).
Interpreters are run with -I -B -u -X utf8. Appends to logs\\swarm2\\verify_E3_3.log.
"""
import os
import subprocess
import time
import traceback

LOG = r"G:\pureskill\csds-cs2\logs\swarm2\verify_E3_3.log"
G_PY = r"G:\pureskill\csds-cs2\.pixi\envs\default\python.exe"
C_PY = r"C:\Users\13144\.pixi\workspace-envs\csds-cs2-15611918389529754563\envs\default\python.exe"
SCRATCH = (r"C:\Users\13144\AppData\Local\Temp\claude"
           r"\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
LINES = []

CODE = r"""
import os, sys, time
print('STAGE 0 exe=' + sys.executable + ' MKL_THREADING_LAYER=' + repr(os.environ.get('MKL_THREADING_LAYER')), flush=True)
import numpy as np
print('STAGE 1 numpy imported ' + np.__version__, flush=True)
rng = np.random.default_rng(0)
a = rng.random((2000, 2000))
t = time.time()
b = a @ a
print('STAGE 2 matmul_ok corner=%.6f sec=%.3f' % (float(b[0, 0]), time.time() - t), flush=True)
import scipy.linalg as sl
print('STAGE 3 scipy imported', flush=True)
m = rng.random((800, 800)) + 800 * np.eye(800)
x = sl.solve(m, np.ones(800))
print('STAGE 4 scipy_solve_ok resid=%.3e' % float(np.abs(m @ x - 1).max()), flush=True)
"""


def out(*parts):
    s = " ".join(str(p) for p in parts)
    LINES.append(s)
    print(s, flush=True)


def run_case(name, py, extra_env, drop=()):
    env = dict(os.environ)
    for k in drop:
        env.pop(k, None)
    env.update(extra_env)
    t = time.time()
    try:
        r = subprocess.run([py, "-I", "-B", "-u", "-X", "utf8", "-c", CODE], capture_output=True,
                           text=True, timeout=240, cwd=SCRATCH, env=env)
        rc = r.returncode
        so = r.stdout.strip()
        se = r.stderr.strip()[-500:]
    except subprocess.TimeoutExpired as e:
        rc = "TIMEOUT"
        so = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        se = "timeout after 240 s"
    out("-- %s rc=%s (0x%08X) elapsed=%.1fs" % (name, rc, rc & 0xFFFFFFFF if isinstance(rc, int) else -1,
                                                time.time() - t))
    for line in so.splitlines():
        out("     stdout: " + line)
    if se:
        out("     stderr tail: " + se.replace("\n", " | "))


def main():
    out("")
    out("== RUNTIME STAGE TRACE (python -u; stage markers) ", time.strftime("%Y-%m-%d %H:%M:%S"))
    run_case("G_default (sitecustomize present)", G_PY, {}, drop=("MKL_THREADING_LAYER",))
    run_case("G_explicit INTEL (sitecustomize default off)", G_PY, {"MKL_THREADING_LAYER": "INTEL"})
    run_case("C_default (no sitecustomize, var unset)", C_PY, {}, drop=("MKL_THREADING_LAYER",))
    run_case("C_candidate_fix MKL_THREADING_LAYER=SEQUENTIAL", C_PY, {"MKL_THREADING_LAYER": "SEQUENTIAL"})
    out("")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        out("EXCEPTION:")
        out(traceback.format_exc())
    finally:
        with open(LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(LINES) + "\n")
