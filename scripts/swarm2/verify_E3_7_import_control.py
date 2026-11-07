"""verify_E3_7_import_control.py -- confounder control for the G: vs C: import gap.

Two confounders found in the env copies before this control:
  (a) bytecode: the C: copy has no .pyc for polars or xgboost (my count, and the envdiff log).
  (b) MKL: G: has sitecustomize.py (17:26) that sets MKL_THREADING_LAYER=SEQUENTIAL by default.

Part 1: numpy 512x512 matmul in each copy, as-is (MKL var removed from the child env) and forced.
Part 2: import timing with bytecode equalized through a scratchpad pycache prefix (no writes
        into either env copy) and MKL_THREADING_LAYER=SEQUENTIAL set explicitly on both.
        ABBA order, two rounds.

Run with: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 -B verify_E3_7_import_control.py
Writes: only the scratchpad pycache prefixes. Env pyc counts are checked before and after.
"""
import os
import statistics
import subprocess
import time

ENV = {
    "G": r"G:\pureskill\csds-cs2\.pixi\envs\default",
    "C": r"C:\Users\13144\.pixi\workspace-envs\csds-cs2-15611918389529754563\envs\default",
}
PY = {k: v + r"\python.exe" for k, v in ENV.items()}
SCRATCH = r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
PREFIX = {k: os.path.join(SCRATCH, "e3_7_pyc_" + k) for k in ENV}

MATMUL = "import numpy as np; a=np.ones((512,512)); b=a@a; print('matmul_ok', float(b[0,0]))"
IMPORTS = "import time;t=time.perf_counter();import numpy,pandas,polars,pyarrow,xgboost,sklearn;print(f'{time.perf_counter()-t:.3f}')"


def pyc_count(root):
    n = 0
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f.endswith(".pyc"):
                n += 1
    return n


def child_env(mkl):
    env = dict(os.environ)
    env.pop("MKL_THREADING_LAYER", None)
    if mkl is not None:
        env["MKL_THREADING_LAYER"] = mkl
    return env


def run(args, env):
    t0 = time.perf_counter()
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    return r, time.perf_counter() - t0


def part1():
    print("== Part 1: numpy matmul in a fresh process (rc shown as unsigned hex)")
    for k in ("G", "C"):
        for label, mkl in (("as_is", None), ("forced_SEQUENTIAL", "SEQUENTIAL")):
            r, secs = run([PY[k], "-I", "-X", "utf8", "-B", "-c", MATMUL], child_env(mkl))
            rc = r.returncode & 0xFFFFFFFF
            out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no stdout)"
            print(f"  {k} {label:18s} rc=0x{rc:08X} secs={secs:.2f} out={out}")


def part2():
    print("== Part 2: bytecode-equalized, MKL forced equal, ABBA x2 (import numpy,pandas,polars,pyarrow,xgboost,sklearn)")
    for k in ("G", "C"):
        os.makedirs(PREFIX[k], exist_ok=True)
    before = {k: pyc_count(ENV[k]) for k in ENV}
    print(f"  env pyc counts before: G={before['G']} C={before['C']}")
    for k in ("G", "C"):
        # warm-up without -B: writes bytecode only into the scratchpad prefix, never into the env
        r, secs = run([PY[k], "-I", "-X", "utf8", "-X", f"pycache_prefix={PREFIX[k]}", "-c",
                       "import numpy,pandas,polars,pyarrow,xgboost,sklearn"], child_env("SEQUENTIAL"))
        print(f"  warm-up {k}: rc={r.returncode} secs={secs:.2f}")
    times = {"G": [], "C": []}
    order = ["G", "C", "C", "G", "G", "C", "C", "G"]
    for k in order:
        r, secs = run([PY[k], "-I", "-X", "utf8", "-B", "-X", f"pycache_prefix={PREFIX[k]}", "-c", IMPORTS],
                      child_env("SEQUENTIAL"))
        inner = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"ERR rc={r.returncode}"
        print(f"  {k}: import_inner_s={inner} wall_s={secs:.2f}")
        try:
            times[k].append(float(inner))
        except ValueError:
            pass
    after = {k: pyc_count(ENV[k]) for k in ENV}
    print(f"  env pyc counts after:  G={after['G']} C={after['C']} (unchanged={before == after})")
    for k in ("G", "C"):
        v = times[k]
        if v:
            print(f"  {k} median={statistics.median(v):.3f} s  min={min(v):.3f} max={max(v):.3f} n={len(v)}")


def main():
    part1()
    part2()
    print("== done")


if __name__ == "__main__":
    main()
