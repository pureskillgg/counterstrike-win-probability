"""verify_E3_7 matmul probe: one numpy BLAS matmul in a fresh process; prints one line.

Tests the sitecustomize.py comment claim (numpy BLAS crashes natively unless MKL uses SEQUENTIAL).
Run by verify_E3_7_import.py. Writes no files (sys.dont_write_bytecode below).
"""
import os
import sys

sys.dont_write_bytecode = True

mkl_before = os.environ.get("MKL_THREADING_LAYER")
import numpy as np  # noqa: E402

a = np.ones((512, 512), dtype=np.float64)
r = a @ a
print("OK sum=%.1f MKL_THREADING_LAYER=%s exe=%s" % (float(r.sum()), mkl_before, sys.executable))
sys.stdout.flush()
