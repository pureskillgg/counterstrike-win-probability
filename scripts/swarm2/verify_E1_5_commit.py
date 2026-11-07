"""verify E1-5: current system commit charge, free RAM, and per-process commit.

Recomputes the key numbers claimed in notes/swarm2/E1.md (E1-5):
  "commit charge 70.7 GB of an 81.2 GB limit; free RAM about 6.3-7.5 GB".
Reads the live values with GetPerformanceInfo (psapi) and per-process private
commit with GetProcessMemoryInfo. This is a snapshot of NOW only; the claimed
values were taken at ~17:06 local and cannot be re-measured retroactively.
Read-only. No writes.

Run:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_E1_5_commit.py
"""
import ctypes
import datetime
import os
from ctypes import wintypes

GB = 2**30


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", wintypes.DWORD),
        ("ProcessCount", wintypes.DWORD),
        ("ThreadCount", wintypes.DWORD),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


psapi = ctypes.WinDLL("psapi")
k32 = ctypes.WinDLL("kernel32")
psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), wintypes.DWORD]
psapi.GetPerformanceInfo.restype = wintypes.BOOL
psapi.EnumProcesses.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
psapi.EnumProcesses.restype = wintypes.BOOL
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k32.OpenProcess.restype = wintypes.HANDLE
k32.CloseHandle.argtypes = [wintypes.HANDLE]
k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
k32.QueryFullProcessImageNameW.restype = wintypes.BOOL


def system_memory():
    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    if not psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        raise OSError("GetPerformanceInfo failed")
    ps = pi.PageSize
    return {
        "commit_total_gb": pi.CommitTotal * ps / GB,
        "commit_limit_gb": pi.CommitLimit * ps / GB,
        "commit_peak_gb": pi.CommitPeak * ps / GB,
        "phys_total_gb": pi.PhysicalTotal * ps / GB,
        "phys_avail_gb": pi.PhysicalAvailable * ps / GB,
        "process_count": pi.ProcessCount,
    }


def process_list():
    n = 4096
    while True:
        arr = (wintypes.DWORD * n)()
        needed = wintypes.DWORD(0)
        if not psapi.EnumProcesses(arr, ctypes.sizeof(arr), ctypes.byref(needed)):
            raise OSError("EnumProcesses failed")
        if needed.value < ctypes.sizeof(arr):
            count = needed.value // ctypes.sizeof(wintypes.DWORD)
            break
        n *= 2
    rows = []
    for pid in arr[:count]:
        if pid == 0:
            continue
        h = k32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED | VM_READ
        if not h:
            continue
        try:
            c = PROCESS_MEMORY_COUNTERS_EX()
            c.cb = ctypes.sizeof(c)
            if not psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                continue
            buf = ctypes.create_unicode_buffer(1024)
            sz = wintypes.DWORD(1024)
            name = ""
            if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(sz)):
                name = buf.value
            rows.append((c.PrivateUsage, c.WorkingSetSize, pid, name))
        finally:
            k32.CloseHandle(h)
    return rows


def main():
    now = datetime.datetime.now().astimezone()
    print("now (local)", now.strftime("%Y-%m-%d %H:%M:%S %Z"), flush=True)
    m = system_memory()
    print(f"commit_total   {m['commit_total_gb']:.2f} GB")
    print(f"commit_limit   {m['commit_limit_gb']:.2f} GB")
    print(f"commit_peak    {m['commit_peak_gb']:.2f} GB (since boot)")
    print(f"commit_ratio   {m['commit_total_gb'] / m['commit_limit_gb']:.3f} (total/limit)")
    print(f"phys_total     {m['phys_total_gb']:.2f} GB")
    print(f"phys_avail     {m['phys_avail_gb']:.2f} GB (free RAM now)")
    print(f"process_count  {m['process_count']}")
    print("")
    claimed_ratio = 70.7 / 81.2
    print(f"claimed (E1.md E1-5, taken ~17:06 local): 70.7 / 81.2 GB = {claimed_ratio:.3f}; free RAM 6.3-7.5 GB")
    print("note: these are live values now; claimed snapshot is not reproducible from here", flush=True)
    print("")
    rows = process_list()
    total_priv = sum(r[0] for r in rows)
    print(f"sum of PrivateUsage over {len(rows)} readable processes: {total_priv / GB:.2f} GB")
    print("top 15 processes by private commit (GB, WS GB, pid, image):")
    for priv, ws, pid, name in sorted(rows, reverse=True)[:15]:
        print(f"  {priv / GB:7.2f}  {ws / GB:6.2f}  {pid:>6}  {name}")
    py = [r for r in rows if os.path.basename(r[3]).lower().startswith("python")]
    print(f"python* processes running now: {len(py)}, private commit sum {sum(r[0] for r in py) / GB:.2f} GB")
    for priv, ws, pid, name in sorted(py, reverse=True):
        print(f"  {priv / GB:7.2f}  {ws / GB:6.2f}  {pid:>6}  {name}")


if __name__ == "__main__":
    main()
