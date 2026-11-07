# -*- coding: utf-8 -*-
"""verify_E3_3_runtime.py

Runtime-equivalence test for finding E3-3. The G: project env has a
sitecustomize.py that sets MKL_THREADING_LAYER=SEQUENTIAL (its comment says
numpy/scipy BLAS calls otherwise crash natively on this machine). The claimed
C: env has no such file. This script runs the same numpy matmul and scipy solve
in each interpreter, as a subprocess, with -I -B (no bytecode is written):

  G_default   G: env python, environment as-is (sitecustomize sets the default)
  G_explicit  G: env python with MKL_THREADING_LAYER=INTEL set explicitly, which
              disables the sitecustomize default (control for the workaround)
  C_default   claimed C: env python, environment as-is (no sitecustomize)

Each case runs 2 times. Results are appended to logs\\swarm2\\verify_E3_3.log.
Writes nothing else (cwd is the scratchpad; -B stops bytecode writes).
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
import numpy as np
print('exe=' + sys.executable)
print('MKL_THREADING_LAYER=' + repr(os.environ.get('MKL_THREADING_LAYER')))
print('numpy=' + np.__version__)
rng = np.random.default_rng(0)
a = rng.random((2000, 2000))
t = time.time()
b = a @ a
print('matmul_ok corner=%.6f sec=%.3f' % (float(b[0, 0]), time.time() - t), flush=True)
import scipy.linalg as sl
m = rng.random((800, 800)) + 800 * np.eye(800)
x = sl.solve(m, np.ones(800))
print('scipy_solve_ok resid=%.3e' % float(np.abs(m @ x - 1).max()), flush=True)
"""


def out(*parts):
    s = " ".join(str(p) for p in parts)
    LINES.append(s)
    print(s, flush=True)


def run_case(name, py, extra_env):
    env = dict(os.environ)
    env.pop("MKL_THREADING_LAYER", None)
    env.update(extra_env)
    t = time.time()
    try:
        r = subprocess.run([py, "-I", "-B", "-X", "utf8", "-c", CODE], capture_output=True,
                           text=True, timeout=240, cwd=SCRATCH, env=env)
        rc = r.returncode
        so = r.stdout.strip()
        se = r.stderr.strip()[-600:]
    except subprocess.TimeoutExpired as e:
        rc = "TIMEOUT"
        so = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        se = "timeout after 240 s"
    out("-- %s (%s) rc=%s elapsed=%.1fs" % (name, os.path.basename(os.path.dirname(os.path.dirname(py))), rc, time.time() - t))
    for line in so.splitlines():
        out("     stdout: " + line)
    if se:
        out("     stderr tail: " + se.replace("\n", " | "))


def main():
    out("")
    out("== RUNTIME EQUIVALENCE (numpy matmul + scipy solve, subprocess, -I -B)", time.strftime("%Y-%m-%d %H:%M:%S"))
    out("host MKL_THREADING_LAYER in this shell: %r" % os.environ.get("MKL_THREADING_LAYER"))
    for rep in (1, 2):
        out("[repeat %d]" % rep)
        run_case("G_default", G_PY, {})
        run_case("G_explicit (sitecustomize default disabled)", G_PY, {"MKL_THREADING_LAYER": "INTEL"})
        run_case("C_default (no sitecustomize)", C_PY, {})
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
