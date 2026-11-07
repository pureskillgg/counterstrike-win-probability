"""E1-1 verifier, part 1: solve-only check of the xgboost CUDA build pin.

Recomputes the build chosen by pixi for three manifests, in a scratch folder
outside the repo (no environment is installed; only pixi.lock is written):

  case A  no [system-requirements], no build pin      (E1 first solve: expected CPU build)
  case B  [system-requirements] cuda = "12.9", no pin (NOT tested in E1 note: the missing cell)
  case C  [system-requirements] cuda = "12.9" + cuda129 pins (E1 claim: expected CUDA build)

Run with the project interpreter (stdlib only):
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_E1_1_solve.py

ASSUMPTION: pixi 0.79.0 at C:\\Users\\13144\\.pixi\\bin\\pixi.exe behaves as in the C: manifest.
"""
import os
import re
import subprocess
import sys

PIXI = r"C:\Users\13144\.pixi\bin\pixi.exe"
SCRATCH_ROOT = (
    r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
    r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_E1_1_solve"
)

COMMON_DEPS = """python = "3.12.*"
polars = ">=2.0.0,<3"
pyarrow = ">=25.0.0,<26"
numpy = ">=2.5.4,<3"
scikit-learn = ">=1.9.1,<2"
pandas = ">=3.0.6,<4"
"""

CASES = {
    "A_nosysreq_nopin": {
        "sysreq": False,
        "xgb": 'xgboost = ">=3.4.2,<4"\n',
    },
    "B_sysreq_nopin": {
        "sysreq": True,
        "xgb": 'xgboost = ">=3.4.2,<4"\n',
    },
    "C_sysreq_pin": {
        "sysreq": True,
        "xgb": (
            'xgboost = { version = "==3.4.2", build = "cuda129*" }\n'
            'libxgboost = { version = "==3.4.2", build = "cuda129*" }\n'
        ),
    },
}


def manifest_text(case):
    spec = CASES[case]
    lines = [
        "[workspace]",
        'channels = ["conda-forge"]',
        f'name = "verify-e1-1-{case.lower()}"',
        'platforms = ["win-64"]',
        'version = "0.1.0"',
        "",
    ]
    if spec["sysreq"]:
        lines += ["[system-requirements]", 'cuda = "12.9"', ""]
    lines += ["[target.win-64.dependencies]", COMMON_DEPS + spec["xgb"]]
    return "\n".join(lines)


def solve(case):
    d = os.path.join(SCRATCH_ROOT, case)
    os.makedirs(d, exist_ok=True)
    manifest = os.path.join(d, "pixi.toml")
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        f.write(manifest_text(case))
    cmd = [PIXI, "lock", "--manifest-path", manifest]
    print(f"== {case}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    print(f"exit={r.returncode}", flush=True)
    tail = "\n".join((r.stderr or "").splitlines()[-6:] + (r.stdout or "").splitlines()[-6:])
    if tail.strip():
        print("pixi output (tail):", flush=True)
        print(tail, flush=True)
    lock = os.path.join(d, "pixi.lock")
    if not os.path.exists(lock):
        print("NO LOCK WRITTEN", flush=True)
        return
    text = open(lock, encoding="utf-8", errors="replace").read()
    platforms = re.findall(r"^- name: (\S+)", text, flags=re.M)
    vpkgs = sorted(set(re.findall(r"^\s+- (__\S+)", text, flags=re.M)))
    xgb_lines = sorted(set(re.findall(r"^- conda: \S*/(xgboost-3\.4\.2-\S+\.conda)", text, flags=re.M)))
    lib_lines = sorted(set(re.findall(r"^- conda: \S*/(libxgboost-3\.4\.2-\S+\.conda)", text, flags=re.M)))
    arrow_cuda = sorted(set(re.findall(r"^- conda: \S*/(libarrow-25\S*_cuda\.conda)", text, flags=re.M)))
    print(f"platform names in lock: {platforms}", flush=True)
    print(f"virtual packages seen in lock: {vpkgs}", flush=True)
    print(f"xgboost packages in lock: {xgb_lines}", flush=True)
    print(f"libxgboost packages in lock: {lib_lines}", flush=True)
    print(f"libarrow *_cuda packages in lock: {len(arrow_cuda)} -> {arrow_cuda}", flush=True)
    cuda_selected = any("cuda129" in x for x in lib_lines)
    print(f"RESULT {case}: libxgboost CUDA build selected = {cuda_selected}", flush=True)


def main():
    print("python", sys.version.split()[0], sys.executable, flush=True)
    print("pixi", PIXI, flush=True)
    for case in CASES:
        solve(case)


if __name__ == "__main__":
    main()
