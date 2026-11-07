"""Busy-loop helper for verify_E2_4_sweep.py (loaded mode). One pure-Python loop per process.

Usage: python -I -X utf8 verify_E2_4_busy.py <seconds>
"""
import sys
import time

end = time.time() + float(sys.argv[1])
while time.time() < end:
    pass
