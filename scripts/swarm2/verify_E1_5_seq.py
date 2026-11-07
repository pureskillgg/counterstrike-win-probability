"""verify_E1_5 sequence repro (independent verifier, own file name). Appends to logs\\swarm2\\verify_E1_5.log.

Parent (project interpreter, G: env) launches two children with the cs2gpu interpreter that the claim used:
  A. the exact claim sequence in ONE process: E1_parity.run(100_000), then E1_parity.run(1_000_000)
  B. a fresh process running only E1_parity.run(1_000_000) (tests whether the earlier 100k step matters)
Each child samples its own commit (PagefileUsage) and the system commit every 2 s, so a crash can be placed
in a phase. The parent records the child's exit code and any new python 0xc000070a app errors.

E1_parity.py is imported unchanged (its __main__ block is not executed).

Run:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -B -X utf8 G:\\pureskill\\csds-cs2\\scripts\\swarm2\\verify_E1_5_seq.py
"""
import ctypes
import datetime as dt
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes

ROOT = r"G:\pureskill\csds-cs2"
SCRIPTS = os.path.join(ROOT, "scripts", "swarm2")
LOG_PATH = os.path.join(ROOT, "logs", "swarm2", "verify_E1_5.log")
CS_PY = r"C:\Users\13144\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default\python.exe"
sys.path.insert(0, SCRIPTS)  # -I drops the script folder from sys.path


class Log:
    def __init__(self, path, mode):
        self.f = open(path, mode, encoding="utf-8", newline="\n")

    def __call__(self, text=""):
        print(text, flush=True)
        self.f.write(text + "\n")
        self.f.flush()


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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
    ]


def proc_mem():
    psapi = ctypes.WinDLL("psapi")
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    c = PROCESS_MEMORY_COUNTERS()
    c.cb = ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return (-1.0, -1.0, -1.0)
    mb = 2 ** 20
    return (c.WorkingSetSize / mb, c.PagefileUsage / mb, c.PeakPagefileUsage / mb)


def child(mode):
    import verify_E1_5_static as S
    import E1_parity as P  # claimed script, unchanged

    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            ws, commit, peak = proc_mem()
            sc = S.system_commit()
            print("[mem %s] proc_ws=%.0fMB proc_commit=%.0fMB proc_peak_commit=%.0fMB | sys_commit=%.2fGiB "
                  "limit=%.2fGiB headroom=%.2fGiB avail_RAM=%.2fGiB" % (
                      time.strftime("%H:%M:%S"), ws, commit, peak, sc["commit_gb"], sc["limit_gb"],
                      sc["limit_gb"] - sc["commit_gb"], sc["avail_ram_gb"]), flush=True)
            stop.wait(2.0)

    threading.Thread(target=sampler, daemon=True).start()
    print("child mode=%s pid=%d interpreter=%s isolated=%d" % (mode, os.getpid(), sys.executable, sys.flags.isolated),
          flush=True)
    print("xgboost", P.xgb.__version__, "USE_CUDA", P.xgb.build_info().get("USE_CUDA"), flush=True)
    if mode == "A":
        print("STEP run(100000) begin", flush=True)
        P.run(100_000)
        print("STEP run(100000) done", flush=True)
    print("STEP run(1000000) begin", flush=True)
    P.run(1_000_000)
    print("STEP run(1000000) done", flush=True)
    stop.set()


def ps_new_crashes(since):
    import verify_E1_5_static as S
    script = (
        "$ProgressPreference = 'SilentlyContinue'\n"
        "$s = [DateTime]'%s'\n"
        "$ev = Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000; StartTime=$s} -ErrorAction SilentlyContinue"
        " | Where-Object { $_.Message -match 'Exception code: 0xc000070a' }\n"
        "foreach ($e in $ev) { $m = $e.Message;"
        " $p = [regex]::Match($m, 'Faulting application path: (.*)').Groups[1].Value.Trim();"
        " $fp = [regex]::Match($m, 'Faulting process id: (\\S+)').Groups[1].Value;"
        " $stt = [regex]::Match($m, 'Faulting application start time: (\\S+)').Groups[1].Value;"
        " $stl = [DateTime]::FromFileTimeUtc([Convert]::ToInt64($stt, 16)).ToLocalTime().ToString('HH:mm:ss');"
        " Write-Output ('NEW|' + $e.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss') + '|' + $p + '|' + $fp + '|' + $stl) }\n"
    ) % since.strftime("%Y-%m-%d %H:%M:%S")
    out, err, rc = S.powershell(script)
    rows = [l.split("|") for l in out.splitlines() if l.startswith("NEW|")]
    return rows, rc, err


def parent():
    import verify_E1_5_static as S
    L = Log(LOG_PATH, "a")
    L("")
    L("## 14. Sequence repro by verify_E1_5_seq.py (own file; the claim's sequence and a fresh 1M-only run)")
    L("Note on section 13 above: during that run the file scripts\\swarm2\\verify_E1_5_repro.py was replaced by other")
    L("content. The child process in section 13 therefore ran the replacement file, not E1_parity run(100k)+run(1M).")
    L("Section 13's child results (nthread 1, 1, 4; 20 trees; G: env) are that other script's output, not the claim's sequence.")
    L("Sections 11 and 12 (site-packages check and the trivial-import controls) were produced by my parent code before the swap.")
    results = {}
    for mode in ("A", "B"):
        before = S.system_commit()
        start = dt.datetime.now()
        proc = subprocess.Popen([CS_PY, "-I", "-B", "-X", "utf8", os.path.abspath(__file__), "--child", mode],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        wd = threading.Timer(900, proc.kill)
        wd.start()
        L("### child %s (%s) started %s pid=%d" % (
            mode, "exact claim sequence, 100k then 1M" if mode == "A" else "fresh process, 1M only", start.strftime("%H:%M:%S"), proc.pid))
        L("    commit before: %.2f GiB of %.2f limit; avail RAM %.2f GiB" % (before["commit_gb"], before["limit_gb"], before["avail_ram_gb"]))
        for line in proc.stdout:
            L("  child%s| %s" % (mode, line.rstrip()))
        rc = proc.wait()
        wd.cancel()
        end = dt.datetime.now()
        after = S.system_commit()
        results[mode] = rc
        L("### child %s exit rc=%d (hex 0x%08X) wall=%.1f s ended %s" % (
            mode, rc, rc & 0xFFFFFFFF, (end - start).total_seconds(), end.strftime("%H:%M:%S")))
        L("    commit after: %.2f GiB of %.2f limit; avail RAM %.2f GiB" % (after["commit_gb"], after["limit_gb"], after["avail_ram_gb"]))
        rows, prc, perr = ps_new_crashes(start - dt.timedelta(seconds=60))
        L("    new python 0xc000070a app errors since child start (-60 s): %d (PowerShell rc=%d)" % (len(rows), prc))
        for r in rows:
            try:
                pid_dec = int(r[3], 16)
            except ValueError:
                pid_dec = -1
            tag = "MATCHES this child" if pid_dec == proc.pid else "other process"
            L("      %s | %s | pid=%s (%d) | start %s | %s" % (r[1], "cs2gpu" if "cs2gpu" in r[2] else "G", r[3], pid_dec, r[4], tag))
        if prc != 0 and perr.strip():
            L("    PowerShell stderr: " + perr.strip()[:200])
    L("")
    L("SEQ DONE: child A rc=%d (hex 0x%08X); child B rc=%d (hex 0x%08X)" % (
        results["A"], results["A"] & 0xFFFFFFFF, results["B"], results["B"] & 0xFFFFFFFF))
    L.f.close()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        parent()
