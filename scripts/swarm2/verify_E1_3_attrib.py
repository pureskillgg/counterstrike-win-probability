"""verify_E1_3 section 5: who is using the host CPU and GPU during an idle window.

No benchmark runs in this process; it only samples counters for WINDOW_S seconds and APPENDS
its result to logs\\swarm2\\verify_E1_3.log (the file written by verify_E1_3_contention.py).
Mandated interpreter:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_E1_3_attrib.py

Method: system busy logical CPUs from GetSystemTimes over the window; per-process CPU-second deltas
from Get-Process for PIDs present in both snapshots (this process excluded). The gap between the two
is kernel/DPC/interrupt time, short-lived processes, and the PowerShell readers themselves.
"""
import ctypes
import os
import subprocess
import time

ROOT = r"G:\pureskill\csds-cs2"
LOG = os.path.join(ROOT, "logs", "swarm2", "verify_E1_3.log")
WINDOW_S = 30  # ASSUMPTION: idle sampling window
NCPU = os.cpu_count() or 16


class FILETIME(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]


_k32 = ctypes.WinDLL("kernel32")


def _ft(f):
    return (f.hi << 32) | f.lo


def sys_times():
    idle, kern, user = FILETIME(), FILETIME(), FILETIME()
    if not _k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
        raise OSError("GetSystemTimes failed")
    return _ft(idle), _ft(kern), _ft(user)


def busy_cores(s0, s1):
    tot = (s1[1] - s0[1]) + (s1[2] - s0[2])
    idle = s1[0] - s0[0]
    return (tot - idle) / tot * NCPU if tot > 0 else float("nan")


def proc_cpu():
    cmd = ("Get-Process | ForEach-Object { '{0}|{1}|{2}' -f $_.Id, $_.ProcessName, "
           "[double]$_.CPU }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                       capture_output=True, text=True, timeout=120)
    d = {}
    for line in r.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3:
            try:
                d[int(parts[0])] = (parts[1], float(parts[2]))
            except ValueError:
                pass
    return d


def gpu_line():
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    return r.stdout.strip()


def gpu_apps():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    return r.stdout.strip() or "(none listed)"


def main():
    lines = []

    def out(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    out("")
    out("== 5. Attribution of host busy time over an idle window (no benchmark in this process) ==")
    out(f"window ASSUMPTION = {WINDOW_S} s; gpu before: {gpu_line()}")
    s_a = sys_times()
    pre = proc_cpu()
    t_a = time.perf_counter()
    time.sleep(WINDOW_S)
    t_b = time.perf_counter()
    post = proc_cpu()
    s_b = sys_times()
    wall = t_b - t_a
    sysb = busy_cores(s_a, s_b)

    me = os.getpid()
    deltas = []
    for pid, (name, cpu1) in post.items():
        if pid == me or pid not in pre:
            continue
        d = cpu1 - pre[pid][1]
        if d > 0:
            deltas.append((d, pid, name))
    deltas.sort(reverse=True)
    proc_total = sum(d for d, _, _ in deltas)
    out(f"system busy logical CPUs over the window (incl. PowerShell readers) = {sysb:.2f} of {NCPU}")
    out(f"sum of per-process CPU deltas (PIDs in both snapshots, this process excluded) = "
        f"{proc_total / wall:.2f} cores (window {wall:.1f} s)")
    out(f"unattributed (kernel/DPC/interrupt, short-lived processes, PowerShell) = {sysb - proc_total / wall:.2f} cores")
    out("top processes by CPU-seconds used in the window:")
    for d, pid, name in deltas[:10]:
        out(f"  {name:<22} pid={pid:<8} cpu_s={d:9.2f}  cores={d / wall:6.2f}")
    out(f"gpu after: {gpu_line()}")
    out(f"gpu compute apps (pid, name, MiB): {gpu_apps()}")
    out("")

    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
