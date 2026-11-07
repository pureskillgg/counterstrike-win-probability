"""verify_E1_4 probe: which interpreter, which OpenMP runtime xgboost loads, OMP-related env vars.

Read-only probe. Writes nothing to disk. Run with the project interpreter:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -B -I -X utf8 scripts\\swarm2\\verify_E1_4_probe.py
or with the C: CUDA env interpreter (same flags).
"""
import ctypes
import os
import sys
from ctypes import wintypes

import numpy as np
import xgboost as xgb


def loaded_modules():
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                         wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    psapi.EnumProcessModules.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE,
                                           wintypes.LPWSTR, wintypes.DWORD]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD
    hproc = kernel32.GetCurrentProcess()
    arr = (wintypes.HMODULE * 4096)()
    needed = wintypes.DWORD()
    ok = psapi.EnumProcessModules(hproc, arr, ctypes.sizeof(arr), ctypes.byref(needed))
    if not ok:
        return []
    n = min(needed.value // ctypes.sizeof(wintypes.HMODULE), len(arr))
    names = []
    buf = ctypes.create_unicode_buffer(520)
    for i in range(n):
        psapi.GetModuleFileNameExW(hproc, arr[i], buf, 520)
        names.append(buf.value)
    return names


def print_omp_modules(label):
    keys = ("omp", "vcomp", "gomp", "iomp", "mkl", "torch", "xgboost", "openblas")
    print(label)
    for m in loaded_modules():
        low = m.lower()
        if any(k in low for k in keys):
            print("  ", m)


def main():
    print("python:", sys.executable)
    print("version:", sys.version.split()[0], "| numpy", np.__version__, "| xgboost", xgb.__version__)
    print("build_info USE_CUDA:", xgb.build_info().get("USE_CUDA"), "USE_OPENMP:", xgb.build_info().get("USE_OPENMP"))
    print("os.cpu_count():", os.cpu_count())
    print("xgboost.__file__:", xgb.__file__)
    print("OMP-related env vars:")
    found = False
    for k in sorted(os.environ):
        if k.upper().startswith(("OMP_", "KMP_", "GOMP_", "MKL_", "OPENBLAS_", "XGBOOST_")):
            print("  ", k, "=", os.environ[k])
            found = True
    if not found:
        print("   (none set)")
    # Train one tiny model so the OpenMP runtime is definitely initialised before listing modules.
    X = np.random.default_rng(0).standard_normal((2000, 5), dtype=np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": "cpu", "nthread": 2},
              xgb.DMatrix(X, label=y), num_boost_round=2)
    print_omp_modules("OpenMP-related modules after numpy + xgboost only:")
    if "--stack" in sys.argv:
        import pandas  # noqa: F401
        import polars  # noqa: F401
        import pyarrow  # noqa: F401
        import sklearn  # noqa: F401
        xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": "cpu", "nthread": 2},
                  xgb.DMatrix(X, label=y), num_boost_round=2)
        print_omp_modules("OpenMP-related modules after adding pandas, polars, pyarrow, sklearn:")


if __name__ == "__main__":
    main()
