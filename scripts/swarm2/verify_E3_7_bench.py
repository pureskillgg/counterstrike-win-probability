"""verify_E3_7_bench.py -- measure what the USB-attached G: volume costs versus internal C:.

Same files, two volumes: the G: copy and the C: detached copy of the csds-cs2 env have
identical binary payloads (see verify_E3_7_envdiff.py). Every file read here exists on both.

Run with: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 -B verify_E3_7_bench.py

Tests (ASSUMPTION list is printed at the top of the output):
  1. sequential unbuffered read of the largest common binary (MB/s), alternating G/C
  2. random 4 KiB unbuffered reads, QD1 (latency percentiles), alternating G/C
  3. working-set pass: read every common file of the import-relevant packages, unbuffered,
     alternating G/C (files/s, MB/s)
  4. interpreter start and import of numpy/pandas/polars/pyarrow/xgboost/sklearn,
     each env's own python, alternating, cold-ish first run and warm runs
  5. tome dataframe bytes on G: (correct walk, all depths)
Read-only. No writes to any volume.
"""
import ctypes
import os
import random
import statistics
import subprocess
import sys
import time
from ctypes import wintypes

GiB = 1024 ** 3
MiB = 1024 ** 2

ENV_G = r"G:\pureskill\csds-cs2\.pixi\envs\default"
ENV_C = r"C:\Users\<user>\.pixi\workspace-envs\csds-cs2-15611918389529754563\envs\default"
PY_G = ENV_G + r"\python.exe"
PY_C = ENV_C + r"\python.exe"
TOMES = [r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds",
         r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"]

WS_PREFIXES = [
    "Lib/site-packages/numpy/", "Lib/site-packages/pandas/", "Lib/site-packages/polars/",
    "Lib/site-packages/_polars_runtime_32/", "Lib/site-packages/pyarrow/",
    "Lib/site-packages/xgboost/", "Lib/site-packages/sklearn/", "Lib/site-packages/scipy/",
    "Lib/site-packages/lightgbm/", "Lib/site-packages/duckdb/", "Library/bin/",
]

GENERIC_READ = 0x80000000
SHARE_ALL = 0x7
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
FILE_BEGIN = 0
MEM_COMMIT_RESERVE = 0x3000
PAGE_READWRITE = 0x04
MEM_RELEASE = 0x8000
ERROR_HANDLE_EOF = 38
INVALID = (1 << 64) - 1

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k32.CreateFileW.restype = wintypes.HANDLE
k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                         ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
k32.ReadFile.restype = wintypes.BOOL
k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                                 ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
k32.SetFilePointerEx.restype = wintypes.BOOL
k32.CloseHandle.argtypes = [wintypes.HANDLE]
k32.CloseHandle.restype = wintypes.BOOL
k32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
k32.VirtualAlloc.restype = wintypes.LPVOID
k32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
k32.VirtualFree.restype = wintypes.BOOL


def log(msg=""):
    print(msg, flush=True)


def open_nobuf(path, sequential):
    flags = FILE_FLAG_NO_BUFFERING | (FILE_FLAG_SEQUENTIAL_SCAN if sequential else FILE_FLAG_RANDOM_ACCESS)
    h = k32.CreateFileW(path, GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, flags, None)
    if h in (None, 0, INVALID):
        raise OSError(ctypes.get_last_error(), "CreateFileW failed", path)
    return h


def round_up(n, a=4096):
    return (n + a - 1) // a * a


def read_whole(h, buf, bufsize, size):
    """Read a whole file unbuffered. Returns logical bytes read (<= size)."""
    need = round_up(size)
    total = 0
    n = wintypes.DWORD(0)
    while total < need:
        req = min(bufsize, need - total)
        ok = k32.ReadFile(h, buf, req, ctypes.byref(n), None)
        if not ok:
            err = ctypes.get_last_error()
            if err == ERROR_HANDLE_EOF:
                break
            raise OSError(err, "ReadFile failed")
        if n.value == 0:
            break
        total += n.value
        if n.value < req:
            break
    return min(total, size)


def alloc(size):
    p = k32.VirtualAlloc(None, size, MEM_COMMIT_RESERVE, PAGE_READWRITE)
    if not p:
        raise OSError(ctypes.get_last_error(), "VirtualAlloc failed")
    return p


def free(p):
    k32.VirtualFree(p, 0, MEM_RELEASE)


def walk_files(root):
    out = {}
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                children = list(it)
        except OSError:
            continue
        for e in children:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                else:
                    out[os.path.relpath(e.path, root).replace("\\", "/")] = e.stat(follow_symlinks=False).st_size
            except OSError:
                pass
    return out


def print_assumptions():
    log("== ASSUMPTIONS")
    log("  A1 FILE_FLAG_NO_BUFFERING reads bypass the Windows system cache, so repeats measure the device path."
        " Drive-internal DRAM cache and NTFS metadata cache still apply.")
    log("  A2 QD1 4 KiB random reads approximate the small-file, latency-bound pattern of Python imports.")
    log("  A3 The G: and C: copies have identical binary payloads (no .dll/.pyd size mismatch in envdiff),"
        " so the same file list is a like-for-like test.")
    log("  A4 Import timing: first run is cold-ish (no prior cache for these files), later runs warm."
        " Warm-run differences are interpreter/CPU effects, not drive effects.")
    log("  A5 Interpreters run with -B, so no .pyc is written into either env. Missing .pyc on C: would"
        " make C: imports slower, so pyc presence is reported per package.")


def seq_read(path, nbytes_cap, chunk=MiB):
    size = os.path.getsize(path)
    want = min(size, nbytes_cap)
    buf = alloc(chunk)
    try:
        h = open_nobuf(path, sequential=True)
        try:
            total = 0
            n = wintypes.DWORD(0)
            t0 = time.perf_counter()
            while total < want:
                req = min(chunk, round_up(want - total))
                ok = k32.ReadFile(h, buf, req, ctypes.byref(n), None)
                if not ok:
                    raise OSError(ctypes.get_last_error(), "ReadFile failed (seq)")
                if n.value == 0:
                    break
                total += n.value
            secs = time.perf_counter() - t0
        finally:
            k32.CloseHandle(h)
    finally:
        free(buf)
    return total, secs


def rand_read(path, reads, seed):
    size = os.path.getsize(path)
    slots = max(1, (size - 4096) // 4096)
    rng = random.Random(seed)
    buf = alloc(4096)
    lat = []
    try:
        h = open_nobuf(path, sequential=False)
        try:
            n = wintypes.DWORD(0)
            for _ in range(reads):
                off = rng.randrange(slots) * 4096
                if not k32.SetFilePointerEx(h, off, None, FILE_BEGIN):
                    raise OSError(ctypes.get_last_error(), "SetFilePointerEx failed")
                t0 = time.perf_counter_ns()
                ok = k32.ReadFile(h, buf, 4096, ctypes.byref(n), None)
                dt = time.perf_counter_ns() - t0
                if not ok:
                    raise OSError(ctypes.get_last_error(), "ReadFile failed (rand)")
                lat.append(dt / 1e6)  # ms
        finally:
            k32.CloseHandle(h)
    finally:
        free(buf)
    return lat


def pct(vals, p):
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def test_seq_and_rand(common_big, big_rel):
    log("== 1. Sequential unbuffered read, largest common binary (alternating G / C)")
    log(f"  file: {big_rel}  size={os.path.getsize(os.path.join(ENV_G, big_rel)) / MiB:.1f} MiB")
    cap = min(os.path.getsize(os.path.join(ENV_G, big_rel)), 512 * MiB)
    seq = {"G": [], "C": []}
    for rep in range(3):
        for tag, root in (("G", ENV_G), ("C", ENV_C)):
            nbytes, secs = seq_read(os.path.join(root, big_rel), cap)
            mbps = nbytes / MiB / secs
            seq[tag].append(mbps)
            log(f"  rep{rep} {tag}: {nbytes / MiB:.1f} MiB in {secs:.3f} s = {mbps:.1f} MiB/s")
    log(f"  median MiB/s: G={statistics.median(seq['G']):.1f}  C={statistics.median(seq['C']):.1f}")

    log("== 2. Random 4 KiB unbuffered reads, QD1 (alternating G / C)")
    rnd = {"G": [], "C": []}
    for rep in range(2):
        for tag, root in (("G", ENV_G), ("C", ENV_C)):
            lat = rand_read(os.path.join(root, big_rel), 300, seed=1000 + rep)
            rnd[tag].extend(lat)
            log(f"  rep{rep} {tag}: n={len(lat)} p50={pct(lat, 50):.3f} ms p95={pct(lat, 95):.3f} ms "
                f"mean={statistics.mean(lat):.3f} ms")
    for tag in ("G", "C"):
        v = rnd[tag]
        log(f"  all {tag}: n={len(v)} p50={pct(v, 50):.3f} ms p95={pct(v, 95):.3f} ms "
            f"p99={pct(v, 99):.3f} ms mean={statistics.mean(v):.3f} ms  (~{1000 / statistics.mean(v):.0f} IOPS QD1)")


def test_working_set(common_rels):
    total_bytes = sum(common_rels.values())
    log("== 3. Working-set pass: every common file under the import-relevant packages, unbuffered")
    log(f"  files={len(common_rels)} logical_GiB={total_bytes / GiB:.3f}")
    chunk = 16 * MiB
    buf = alloc(chunk)
    results = {"G": [], "C": []}
    try:
        for rep in range(2):
            for tag, root in (("G", ENV_G), ("C", ENV_C)):
                got_bytes = 0
                short = 0
                errors = 0
                t0 = time.perf_counter()
                for rel in sorted(common_rels):
                    size = common_rels[rel]
                    if size == 0:
                        continue
                    try:
                        h = open_nobuf(os.path.join(root, rel.replace("/", "\\")), sequential=True)
                        try:
                            n = read_whole(h, buf, chunk, size)
                        finally:
                            k32.CloseHandle(h)
                        got_bytes += n
                        if n != size:
                            short += 1
                    except OSError:
                        errors += 1
                secs = time.perf_counter() - t0
                results[tag].append(secs)
                log(f"  rep{rep} {tag}: secs={secs:.2f} files/s={len(common_rels) / secs:.0f} "
                    f"MiB/s={got_bytes / MiB / secs:.1f} short_reads={short} errors={errors}")
    finally:
        free(buf)
    log(f"  median secs: G={statistics.median(results['G']):.2f}  C={statistics.median(results['C']):.2f}")


def test_python_start_and_import():
    log("== 4. Interpreter start and import (each env's own python, -I -X utf8 -B)")
    code_start = "pass"
    code_imp = ("import time;t=time.perf_counter();"
                "import numpy,pandas,polars,pyarrow,xgboost,sklearn;"
                "print(f'{time.perf_counter()-t:.3f}')")
    starts = {"G": [], "C": []}
    for rep in range(3):
        for tag, py in (("G", PY_G), ("C", PY_C)):
            t0 = time.perf_counter()
            subprocess.run([py, "-I", "-X", "utf8", "-B", "-c", code_start], capture_output=True, text=True)
            starts[tag].append(time.perf_counter() - t0)
    log(f"  bare start median s: G={statistics.median(starts['G']):.3f} C={statistics.median(starts['C']):.3f}"
        f" (runs G={[round(x, 3) for x in starts['G']]} C={[round(x, 3) for x in starts['C']]})")
    imps = {"G": [], "C": []}
    for rep in range(3):
        for tag, py in (("G", PY_G), ("C", PY_C)):
            t0 = time.perf_counter()
            r = subprocess.run([py, "-I", "-X", "utf8", "-B", "-c", code_imp], capture_output=True, text=True)
            wall = time.perf_counter() - t0
            inner = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"ERR rc={r.returncode} {r.stderr[-300:]}"
            log(f"  rep{rep} {tag}: import_inner_s={inner} wall_s={wall:.2f}")
            try:
                imps[tag].append(float(inner))
            except ValueError:
                pass
    for tag in ("G", "C"):
        if imps[tag]:
            log(f"  {tag} import median s={statistics.median(imps[tag]):.3f} first={imps[tag][0]:.3f}")


def pyc_presence(g_map, c_map):
    log("== 4b. .pyc presence for the imported packages (affects -B import timing)")
    for pkg in ("numpy", "pandas", "polars", "_polars_runtime_32", "pyarrow", "xgboost", "sklearn"):
        pre = f"Lib/site-packages/{pkg}/"
        py_g = [k for k in g_map if k.startswith(pre) and k.endswith(".py")]
        pyc_g = [k for k in g_map if k.startswith(pre) and k.endswith(".pyc")]
        py_c = [k for k in c_map if k.startswith(pre) and k.endswith(".py")]
        pyc_c = [k for k in c_map if k.startswith(pre) and k.endswith(".pyc")]
        log(f"  {pkg:20s} G: py={len(py_g)} pyc={len(pyc_g)}   C: py={len(py_c)} pyc={len(pyc_c)}")


def tome_totals():
    log("== 5. Tome dataframe bytes on G: (all depths)")
    for root in TOMES:
        n = 0
        total = 0
        t0 = time.perf_counter()
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if f.startswith("dataframe_"):
                    n += 1
                    total += os.path.getsize(os.path.join(dp, f))
        log(f"  {root}: dataframe files={n} GiB={total / GiB:.3f} walk_s={time.perf_counter() - t0:.1f}")


def main():
    print_assumptions()
    log("== building common file lists")
    g_all = walk_files(ENV_G)
    c_all = walk_files(ENV_C)
    common = {k: v for k, v in g_all.items() if k in c_all and c_all[k] == v}
    ws = {k: v for k, v in common.items() if k.startswith(tuple(WS_PREFIXES))}
    big_candidates = [(v, k) for k, v in common.items() if k.endswith((".pyd", ".dll")) and v >= 16 * MiB]
    big_candidates.sort(reverse=True)
    big_rel = big_candidates[0][1] if big_candidates else max(common, key=common.get)
    log(f"  G files={len(g_all)} C files={len(c_all)} common_same_size={len(common)} working_set={len(ws)}")
    test_seq_and_rand(common, big_rel)
    test_working_set(ws)
    pyc_presence(g_all, c_all)
    test_python_start_and_import()
    tome_totals()
    log("== done")


if __name__ == "__main__":
    main()
