"""verify_E3_7 data-path probe (read-only): time reading every tome data file on G:, plus free space.

The task context said tome data is dataframe_* parquet. The extension census in verify_E3_7_pyc.py
found no .parquet files; the data files are extensionless. This probe reads every file under both
G: tome roots (pass 1, then pass 2) and reports bytes and seconds. Nothing is written.

Run:
  G:/pureskill/csds-cs2/.pixi/envs/default/python.exe -I -X utf8 G:/pureskill/csds-cs2/scripts/swarm2/verify_E3_7_tome_read.py
"""
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True

GIB = 1024.0 ** 3
MIB = 1024.0 ** 2
CHUNK = 1 << 20
TOMES = [
    "G:\\pureskill\\csds-cs2\\tomes\\2026-10-01_to_08\\tome\\csds",
    "G:\\pureskill\\csds-cs2\\tomes\\2024-02-g\\tome\\csds",
]


def out(s=""):
    print(s, flush=True)


def list_files(root):
    paths = []
    for dp, dns, fns in os.walk(root):
        for fn in fns:
            paths.append(os.path.join(dp, fn))
    return paths


def read_all(paths):
    total = 0
    errors = 0
    t0 = time.perf_counter()
    for p in paths:
        try:
            with open(p, "rb") as fh:
                while True:
                    b = fh.read(CHUNK)
                    if not b:
                        break
                    total += len(b)
        except OSError:
            errors += 1
    return time.perf_counter() - t0, total, errors


def main():
    out("verify_E3_7_tome_read.py  interpreter=%s" % sys.executable)
    out("== 1. Free space (shutil.disk_usage, measured now) ==")
    for letter in ["C:", "E:", "F:", "G:"]:
        u = shutil.disk_usage(letter + "\\")
        out("  %s total=%.2f GiB free=%.2f GiB" % (letter, u.total / GIB, u.free / GIB))
    out("== 2. Tome data read on G: (every file, extensionless data files) ==")
    for root in TOMES:
        paths = list_files(root)
        out("  root=%s files=%d" % (root, len(paths)))
        s1, b1, e1 = read_all(paths)
        out("    pass1 bytes=%.1f MiB secs=%.3f MiB/s=%.1f errors=%d" % (b1 / MIB, s1, (b1 / MIB) / s1 if s1 else 0.0, e1))
        s2, b2, e2 = read_all(paths)
        out("    pass2 bytes=%.1f MiB secs=%.3f MiB/s=%.1f errors=%d" % (b2 / MIB, s2, (b2 / MIB) / s2 if s2 else 0.0, e2))
    out("== done (read-only) ==")


if __name__ == "__main__":
    main()
