"""Reporting machinery shared by the check scripts (e2e_checks.py, parity_check.py, test_snapshot_parity.py).

Severity levels
  PASS     the invariant holds on a non-empty population (every check carries the size n of what it looked at).
  WARN     a documented known limitation, measured and printed with its number; never fails the run. A check may only return
           WARN where the limitation is understood and a threshold (or a baseline file) bounds how bad it may get.
  FAIL     the invariant is violated, the population is empty (a vacuous pass), or the check raised (the traceback is recorded).
  NOT RUN  the check needs a file or a model that does not exist. Counts as FAIL unless --allow-missing is given.

The exit code of a script is 1 on any FAIL (or any NOT RUN without --allow-missing); WARN never fails.
The JSON report is written atomically (temp file + os.replace) and also when a check raises; a "running" report with
all_passed=false is written first, so a crashed run can never leave an older all_passed=true file standing.
"""
import hashlib
import json
import os
import sys
import time
import traceback

PASS, WARN, FAIL, NOTRUN, SKIP = "PASS", "WARN", "FAIL", "NOT RUN", "SKIP"


def project_root():
    """scripts/model/checks/core.py -> the project root (three directories above checks/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


# ------------------------------------------------------------------ results
class Res:
    __slots__ = ("status", "detail", "n", "data")

    def __init__(self, status, detail="", n=None, data=None):
        self.status, self.detail, self.n, self.data = status, str(detail), n, data


def ok(detail="", n=None, **data):
    return Res(PASS, detail, n, data or None)


def warn(detail="", n=None, **data):
    return Res(WARN, detail, n, data or None)


def fail(detail="", n=None, **data):
    return Res(FAIL, detail, n, data or None)


def notrun(detail=""):
    return Res(NOTRUN, detail)


def skip(detail=""):
    """Deliberately not run (--fast): neither a pass nor a failure."""
    return Res(SKIP, detail)


def verdict(cond, detail="", n=None, **data):
    return Res(PASS if cond else FAIL, detail, n, data or None)


def _coerce(out):
    if isinstance(out, Res):
        return out
    if isinstance(out, bool):
        return Res(PASS if out else FAIL)
    if isinstance(out, tuple) and out and isinstance(out[0], bool):
        return Res(PASS if out[0] else FAIL, out[1] if len(out) > 1 else "", out[2] if len(out) > 2 else None)
    raise TypeError(f"a check must return a Res, a bool or (bool, detail[, n]); got {type(out).__name__}")


class Reporter:
    def __init__(self, allow_missing=False, echo=True, default_min_n=1):
        self.allow_missing = allow_missing
        self.echo = echo
        self.default_min_n = default_min_n
        self.results = []

    # -- recording
    def add(self, cid, name, res, min_n=None, seconds=None, tb=None):
        res = _coerce(res)
        min_n = self.default_min_n if min_n is None else min_n
        status, detail = res.status, res.detail
        if status in (PASS, WARN) and res.n is not None and res.n < min_n:
            status, detail = FAIL, f"VACUOUS: population n={res.n} < {min_n}; {detail}"
        counts_as_fail = status == FAIL or (status == NOTRUN and not self.allow_missing)
        label = "FAIL" if counts_as_fail else status
        if status == NOTRUN:
            detail = ("NOT RUN: " + detail) if counts_as_fail else detail
        rec = {"id": cid, "check": name, "status": status, "passed": not counts_as_fail, "detail": detail[:600],
               "n": None if res.n is None else int(res.n), "peak_rss_mb": peak_rss_mb()}
        if res.data:
            rec["data"] = res.data
        if seconds is not None:
            rec["seconds"] = round(seconds, 2)
        if tb:
            rec["traceback"] = tb
        self.results.append(rec)
        if self.echo:
            print(f"{label:<8s}{cid} {name}  {detail[:400]}", flush=True)
        return rec

    def run(self, cid, name, fn, min_n=None):
        """Run one check; an exception is recorded as FAIL with the traceback and never stops the run."""
        only = getattr(self, "only", None)
        if only is not None and not any(cid.startswith(p) for p in only):
            return None
        t0 = time.time()
        tb = None
        try:
            res = _coerce(fn())
        except Exception as exc:  # noqa: BLE001 - the contract: a broken check is a FAIL, not a crash
            tb = traceback.format_exc()
            res = Res(FAIL, f"check raised {type(exc).__name__}: {str(exc)[:300]}")
        return self.add(cid, name, res, min_n=min_n, seconds=time.time() - t0, tb=tb)

    def skip(self, cids, why):
        """Record NOT RUN for a group of checks that miss a prerequisite. cids: list of (id, name)."""
        for cid, name in cids:
            self.add(cid, name, notrun(why))

    # -- summary
    def failed(self):
        return [r for r in self.results if not r["passed"]]

    def warned(self):
        return [r for r in self.results if r["status"] == WARN]

    def counts(self):
        c = {}
        for r in self.results:
            c[r["status"]] = c.get(r["status"], 0) + 1
        return c

    def all_passed(self):
        return bool(self.results) and not self.failed()

    def summary_line(self):
        c = self.counts()
        failed = [r["id"] for r in self.failed()]
        txt = ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
        return ("RESULT: ALL PASS" if not failed else f"RESULT: FAILED {failed}") + f"  ({txt})"


# ------------------------------------------------------------------ parameters
class Params:
    """Every threshold is a parameter with a default; --param name=value overrides one. Unknown names are an error (a typo must not
    silently leave the default in place)."""

    def __init__(self, defaults, overrides=None):
        self._d = dict(defaults)
        for item in overrides or []:
            k, _, v = str(item).partition("=")
            k = k.strip()
            if k not in self._d:
                raise KeyError(f"unknown parameter {k!r}; known: {sorted(self._d)}")
            cur = self._d[k]
            if isinstance(cur, bool):
                self._d[k] = v.strip().lower() in ("1", "true", "yes", "y")
            elif isinstance(cur, int):
                self._d[k] = int(float(v))
            elif isinstance(cur, float):
                self._d[k] = float(v)
            else:
                self._d[k] = v

    def __getattr__(self, k):
        try:
            return self.__dict__["_d"][k]
        except KeyError as e:
            raise AttributeError(k) from e

    def as_dict(self):
        return dict(self._d)


# ------------------------------------------------------------------ files, hashes, memory
def jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def dump_json_atomic(obj, path):
    """Temp file in the same folder, then os.replace: an interrupted run never leaves a truncated or half-updated report."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=jsonable)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        left = limit
        while True:
            b = f.read(1 << 20 if left is None else min(1 << 20, left))
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
                if left <= 0:
                    break
    return h.hexdigest()


def sha1_first_mib(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read(1 << 20)).hexdigest()


def file_info(path, sha_max_mb=64.0, rows=False):
    """Manifest entry: size, mtime, sha256 (full hash for files up to sha_max_mb, otherwise of the first and last MiB with the size),
    and for a parquet file the row count."""
    try:
        st = os.stat(path)
    except OSError:
        return {"path": path, "exists": False}
    info = {"path": path, "exists": True, "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))}
    if st.st_size <= sha_max_mb * 2 ** 20:
        info["sha256"] = sha256_file(path)
    else:
        with open(path, "rb") as f:
            head = f.read(1 << 20)
            f.seek(max(0, st.st_size - (1 << 20)))
            tail = f.read(1 << 20)
        info["sha256_head_tail_1mib"] = hashlib.sha256(head + tail + str(st.st_size).encode()).hexdigest()
    if rows and path.endswith(".parquet"):
        try:
            import polars as pl
            info["rows"] = int(pl.scan_parquet(path).select(pl.len()).collect().item())
        except Exception as exc:  # noqa: BLE001
            info["rows_error"] = type(exc).__name__
    return info


def list_parquets(directory, min_age_s=120.0):
    """Sorted *.parquet files of a folder, leaving out files written in the last min_age_s seconds (an extractor may still be writing them)."""
    import glob
    now = time.time()
    out = []
    for f in sorted(glob.glob(os.path.join(directory, "*.parquet"))):
        try:
            if now - os.path.getmtime(f) >= min_age_s:
                out.append(f)
        except OSError:
            continue
    return out


def peak_rss_mb():
    """Peak working set of this process in MiB (None where unavailable)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)
            return None
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:  # noqa: BLE001
        return None


def setup_env(threads=2):
    """Call before importing polars / numpy: sequential MKL, a small polars pool."""
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("POLARS_MAX_THREADS", str(threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
