"""verify_E3_7_blas_nosite.py -- is the C: BLAS crash a placement effect or a sitecustomize effect?

Each copy's python is started with -S (site not run, so sitecustomize.py is NOT loaded on either
copy). site-packages and the env's Library/bin are added by hand, identically for both copies.
Then a numpy 512x512 matmul is run with MKL_THREADING_LAYER removed from the child environment.

Run with: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 -B verify_E3_7_blas_nosite.py
Read-only on both env copies; no bytecode written (-B).
"""
import os
import subprocess

ENV = {
    "G": r"G:\pureskill\csds-cs2\.pixi\envs\default",
    "C": r"C:\Users\13144\.pixi\workspace-envs\csds-cs2-15611918389529754563\envs\default",
}

CODE = (
    "import os, sys\n"
    "root = sys.prefix\n"
    "sp = os.path.join(root, 'Lib', 'site-packages')\n"
    "sys.path.insert(0, sp)\n"
    "bindir = os.path.join(root, 'Library', 'bin')\n"
    "if hasattr(os, 'add_dll_directory') and os.path.isdir(bindir):\n"
    "    os.add_dll_directory(bindir)\n"
    "import numpy as np\n"
    "a = np.ones((512, 512)); b = a @ a\n"
    "print('matmul_ok', float(b[0, 0]), 'sitecustomize_loaded=', 'sitecustomize' in sys.modules)\n"
)


def main():
    env_base = dict(os.environ)
    env_base.pop("MKL_THREADING_LAYER", None)
    for k in ("G", "C"):
        py = ENV[k] + r"\python.exe"
        r = subprocess.run([py, "-I", "-X", "utf8", "-B", "-S", "-c", CODE],
                           capture_output=True, text=True, env=env_base)
        rc = r.returncode & 0xFFFFFFFF
        out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no stdout)"
        err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        print(f"  {k} -S as-is: rc=0x{rc:08X} out={out} stderr_tail={err[:160]!r}")
    print("== done")


if __name__ == "__main__":
    main()
