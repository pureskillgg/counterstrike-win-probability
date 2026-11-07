"""verify_E1_5 static checks (independent verifier). No training runs in this script.

Recomputes, from raw artifacts only:
  - E1_parity.log / E1_bench.log / E1_diag.log facts (1M step presence, peaks, errors)
  - Windows Application and System event records for python 0xc000070a crashes and
    low-virtual-memory events (via PowerShell, read-only)
  - WER report metadata for the 17:06 to 17:08 crashes (read-only)
  - NTSTATUS 0xC000070A text (FormatMessage against ntdll) and the nearest ntdll export to the
    claimed fault offset 0xC87A4 (PE export table parse)
  - current system commit charge, commit limit and available RAM (GetPerformanceInfo)

Writes (overwrites): logs\\swarm2\\verify_E1_5.log
verify_E1_5_repro.py appends to the same log.

Run:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -B -X utf8 G:\\pureskill\\csds-cs2\\scripts\\swarm2\\verify_E1_5_static.py
"""
import base64
import ctypes
import datetime as dt
import json
import os
import re
import struct
import subprocess
import sys
from ctypes import wintypes

ROOT = r"G:\pureskill\csds-cs2"
LOGS = os.path.join(ROOT, "logs", "swarm2")
LOG_PATH = os.path.join(LOGS, "verify_E1_5.log")
WER_ARCHIVE = r"C:\ProgramData\Microsoft\Windows\WER\ReportArchive"
NTDLL = r"C:\Windows\System32\ntdll.dll"
CLAIM_OFFSET = 0xC87A4
CLAIM_CODE = 0xC000070A
G_PY = r"G:\pureskill\csds-cs2\.pixi\envs\default\python.exe"
CS_PY = r"C:\Users\<user>\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default\python.exe"
GB = 2 ** 30


class Log:
    def __init__(self, path, mode):
        self.f = open(path, mode, encoding="utf-8", newline="\n")

    def __call__(self, text=""):
        print(text, flush=True)
        self.f.write(text + "\n")
        self.f.flush()


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def file_line(path):
    st = os.stat(path)
    return ("%s: size=%d bytes, created=%s, modified=%s" % (
        os.path.basename(path), st.st_size,
        dt.datetime.fromtimestamp(st.st_ctime).strftime("%H:%M:%S"),
        dt.datetime.fromtimestamp(st.st_mtime).strftime("%H:%M:%S")))


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


def system_commit():
    psapi = ctypes.WinDLL("psapi")
    psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), wintypes.DWORD]
    psapi.GetPerformanceInfo.restype = wintypes.BOOL
    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    if not psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        raise OSError(ctypes.get_last_error())
    ps = pi.PageSize
    return {
        "commit_gb": pi.CommitTotal * ps / GB,
        "limit_gb": pi.CommitLimit * ps / GB,
        "peak_gb": pi.CommitPeak * ps / GB,
        "avail_ram_gb": pi.PhysicalAvailable * ps / GB,
        "total_ram_gb": pi.PhysicalTotal * ps / GB,
    }


def ntstatus_text(code):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    k32.GetModuleHandleW.restype = wintypes.HMODULE
    k32.FormatMessageW.argtypes = [wintypes.DWORD, wintypes.HMODULE, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.LPWSTR, wintypes.DWORD, ctypes.c_void_p]
    k32.FormatMessageW.restype = wintypes.DWORD
    hmod = k32.GetModuleHandleW("ntdll.dll")
    buf = ctypes.create_unicode_buffer(2048)
    n = k32.FormatMessageW(0x800 | 0x1000 | 0x200, hmod, code, 0, buf, 2048, None)
    if not n:
        return "(no message text; GetLastError=%d)" % ctypes.get_last_error()
    return buf.value.strip()


def ntdll_nearest_export(path, target):
    data = open(path, "rb").read()
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[e_lfanew:e_lfanew + 4] == b"PE\x00\x00", "bad PE signature"
    nsec = struct.unpack_from("<H", data, e_lfanew + 6)[0]
    opt_size = struct.unpack_from("<H", data, e_lfanew + 20)[0]
    opt = e_lfanew + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    exp_rva = struct.unpack_from("<I", data, dd)[0]
    sec_off = opt + opt_size
    sections = []
    for i in range(nsec):
        o = sec_off + 40 * i
        vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", data, o + 8)
        sections.append((va, max(vsize, rawsize), rawptr))

    def r2o(rva):
        for va, size, rawptr in sections:
            if va <= rva < va + size:
                return rva - va + rawptr
        raise ValueError("rva not in sections: %x" % rva)

    ed = r2o(exp_rva)
    nfunc = struct.unpack_from("<I", data, ed + 20)[0]
    nnames = struct.unpack_from("<I", data, ed + 24)[0]
    aof = struct.unpack_from("<I", data, ed + 28)[0]
    aon = struct.unpack_from("<I", data, ed + 32)[0]
    aono = struct.unpack_from("<I", data, ed + 36)[0]
    funcs = [struct.unpack_from("<I", data, r2o(aof) + 4 * i)[0] for i in range(nfunc)]
    named = []
    for i in range(nnames):
        name_rva = struct.unpack_from("<I", data, r2o(aon) + 4 * i)[0]
        idx = struct.unpack_from("<H", data, r2o(aono) + 2 * i)[0]
        off = r2o(name_rva)
        end = data.index(b"\x00", off)
        named.append((funcs[idx], data[off:end].decode("ascii", "replace")))
    below_named = max(((f, n) for f, n in named if 0 < f <= target), default=None)
    below_any = max((f for f in funcs if 0 < f <= target), default=None)
    return len(data), below_named, below_any, len(named)


def powershell(ps_text, timeout=300):
    enc = base64.b64encode(ps_text.encode("utf-16-le")).decode("ascii")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", enc],
                       capture_output=True, timeout=timeout)
    return r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace"), r.returncode


PS_EVENTS = r"""
$start = (Get-Date).AddDays(-5)
$ap = Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000; StartTime=$start} -ErrorAction SilentlyContinue | Where-Object { $_.Message -match 'Exception code: 0xc000070a' }
foreach ($e in $ap) {
  $m = $e.Message
  $p = [regex]::Match($m, 'Faulting application path: (.*)').Groups[1].Value.Trim()
  $fp = [regex]::Match($m, 'Faulting process id: (\S+)').Groups[1].Value
  $ts = [regex]::Match($m, 'time stamp: (\S+)').Groups[1].Value
  $off = [regex]::Match($m, 'Fault offset: (\S+)').Groups[1].Value
  $stt = [regex]::Match($m, 'Faulting application start time: (\S+)').Groups[1].Value
  $stl = [DateTime]::FromFileTimeUtc([Convert]::ToInt64($stt, 16)).ToLocalTime().ToString('HH:mm:ss')
  Write-Output ('APP|' + $e.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss') + '|' + $p + '|' + $fp + '|' + $ts + '|' + $off + '|' + $stl)
}
$w = Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1001; StartTime=[DateTime]'2026-10-13 17:05:00'; EndTime=[DateTime]'2026-10-13 17:09:00'} -ErrorAction SilentlyContinue
foreach ($e in $w) {
  $m = $e.Message
  $p1 = [regex]::Match($m, 'P1: (\S+)').Groups[1].Value
  $p3 = [regex]::Match($m, 'P3: (\S+)').Groups[1].Value
  $p7 = [regex]::Match($m, 'P7: (\S+)').Groups[1].Value
  $p8 = [regex]::Match($m, 'P8: (\S+)').Groups[1].Value
  Write-Output ('WER|' + $e.TimeCreated.ToString('HH:mm:ss') + '|' + $p1 + '|' + $p3 + '|' + $p7 + '|' + $p8)
}
$lv = Get-WinEvent -FilterHashtable @{LogName='System'; Id=2004; StartTime=$start} -ErrorAction SilentlyContinue
foreach ($e in $lv) {
  Write-Output ('LOWVM|' + $e.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss') + '|' + (($e.Message -split '\r?\n') -join ' '))
}
"""


def wer_reports(t0, t1):
    out = []
    if not os.path.isdir(WER_ARCHIVE):
        return ["WER archive missing"]
    for name in sorted(os.listdir(WER_ARCHIVE)):
        if "python" not in name:
            continue
        full = os.path.join(WER_ARCHIVE, name)
        try:
            mtime = dt.datetime.fromtimestamp(os.stat(full).st_mtime)
        except OSError:
            continue
        if not (t0 <= mtime <= t1):
            continue
        rep = os.path.join(full, "Report.wer")
        try:
            raw = open(rep, "rb").read().decode("utf-16", "replace")
        except OSError as e:
            out.append("%s modified=%s Report.wer unreadable (%s)" % (name[:40], mtime.strftime("%H:%M:%S"), type(e).__name__))
            continue
        guid = re.search(r"AppSessionGuid=(\S+)", raw)
        ticks = re.search(r"EventTime=(\d+)", raw)
        app = re.search(r"AppPath=(.*)", raw)
        ev_local = "?"
        if ticks:
            ev_utc = dt.datetime(1601, 1, 1) + dt.timedelta(microseconds=int(ticks.group(1)) // 10)
            ev_local = ev_utc.replace(tzinfo=dt.timezone.utc).astimezone().strftime("%H:%M:%S")
        out.append("modified=%s EventTime(local)=%s AppSessionGuid=%s AppPath=%s" % (
            mtime.strftime("%H:%M:%S"), ev_local, guid.group(1) if guid else "?", app.group(1).strip() if app else "?"))
    return out


def main():
    L = Log(LOG_PATH, "w")
    L("# verify_E1_5 log (independent verifier)")
    L("Claim under test: E1-5 'memory/commit pressure caused a CPU bad allocation and native crashes', verdict 'supported' (notes\\swarm2\\E1.md).")
    L("Interpreter: %s" % sys.executable)
    L("sys.flags.isolated=%d  python=%s" % (sys.flags.isolated, sys.version.split()[0]))
    L("Run time (local): %s" % dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    L("")

    L("## 1. Artifact timestamps (creation and modification, local)")
    for p in [os.path.join(LOGS, "E1_bench.log"), os.path.join(LOGS, "E1_parity.log"),
              os.path.join(LOGS, "E1_diag.log"), os.path.join(LOGS, "E1_threads.log"),
              os.path.join(ROOT, "scripts", "swarm2", "E1_parity.py"),
              os.path.join(ROOT, "scripts", "swarm2", "E1_bench.py"),
              os.path.join(ROOT, "notes", "swarm2", "E1.md")]:
        L("  " + file_line(p))
    L("  ASSUMPTION: created time on Windows = file creation time (Python st_ctime).")
    L("")

    L("## 2. E1_parity.log content (the 1M step of the claim)")
    par = read_text(os.path.join(LOGS, "E1_parity.log"))
    n100 = len(re.findall(r"n_train=\s*100,000", par))
    n1m = len(re.findall(r"n_train=\s*1,000,000", par))
    L("  lines with n_train=100,000: %d ; lines with n_train=1,000,000: %d" % (n100, n1m))
    L("  'Traceback' present: %s ; 'bad allocation' present: %s" % ("Traceback" in par, "bad allocation" in par))
    L("  E1_parity.py run order (from source): run(100_000) then run(1_000_000) in ONE process (verified by reading the file).")
    L("  => the log ends after 100k output; the 1M step left no text. The 1M crash is known only from Windows events.")
    L("")

    L("## 3. E1_bench.log (completed 1M runs, final log)")
    bench = read_text(os.path.join(LOGS, "E1_bench.log"))
    m = re.search(r"SUMMARY_JSON_BEGIN\s*(\{.*\})\s*SUMMARY_JSON_END", bench, re.S)
    js = json.loads(m.group(1))
    res = js["results"]
    L("  workload n_train=%d n_features=%d trees=%d depth=%d" % (
        js["workload"]["n_train"], js["workload"]["n_features"], js["workload"]["n_trees"], js["workload"]["max_depth"]))
    for k in ("cpu_default", "cpu_1thread", "cuda"):
        L("  %-12s train_times_s=%s median=%.3f holdout_logloss=%s peak_rss_mb_so_far=%s" % (
            k, res[k]["train_times_s"], res[k]["train_median_s"], res[k]["holdout_logloss"], res[k]["peak_rss_mb_so_far"]))
    ws_peaks = [float(x.group(2)) for x in re.finditer(r"ws,peak_MB=\(([\d.]+), ([\d.]+)\)", bench)]
    L("  max peak working set logged anywhere in E1_bench.log: %.1f MB (%d samples)" % (max(ws_peaks), len(ws_peaks)))
    L("  'bad allocation' in final E1_bench.log: %s ; 'XGBoostError' in final E1_bench.log: %s" % (
        "bad allocation" in bench, "XGBoostError" in bench))
    L("  ASSUMPTION: the first (overwritten) attempt is not recoverable; E1_bench.py mtime is after its creation.")
    L("")

    L("## 4. E1_diag.log (row-scaling diagnostic, same features, 20 trees)")
    diag = read_text(os.path.join(LOGS, "E1_diag.log")).splitlines()
    for line in diag:
        if line.startswith("n="):
            n = re.match(r"n=\s*([\d,]+)", line).group(1)
            status = "OK" if " OK " in line else "FAILED"
            pk = re.search(r"peak=([\d.]+)MB", line).group(1)
            dev = re.search(r"device=(\w+)", line).group(1)
            nt = re.search(r"nthread=(\S+)", line).group(1)
            L("  %-14s device=%-5s nthread=%-5s status=%-6s peak_ws_MB=%s" % (n, dev, nt, status, pk))
    L("  order in one process: 100k cpu, 300k cpu, 1M cpu, 1M cpu nthread=4, 1M cuda (same pattern as E1_parity: smaller first, then 1M).")
    L("")

    L("## 5. NTSTATUS text and arithmetic")
    t_claim = ntstatus_text(CLAIM_CODE)
    L("  0xC000070A (claim code) text: %s" % t_claim)
    if t_claim.startswith("Status 0x"):
        L("  -> generic template only: ntdll's message table has no public text for 0xC000070A (facility 0, code 0x70A).")
        L("     It is not the standard low-memory status (0xC0000017 / 0xC000012D); a commit-limit failure usually surfaces as those.")
    L("  -1073740022 & 0xFFFFFFFF = %s (claim exit code)" % hex(-1073740022 & 0xFFFFFFFF))
    for c in (0xC0000017, 0xC000012D, 0xC0000005, 0xC0000409):
        L("  for comparison %s: %s" % (hex(c), ntstatus_text(c)))
    L("")

    L("## 6. ntdll.dll nearest export to the claimed fault offset 0xC87A4")
    size, below_named, below_any, nnamed = ntdll_nearest_export(NTDLL, CLAIM_OFFSET)
    L("  on-disk ntdll.dll size=%d bytes; named exports=%d" % (size, nnamed))
    if below_named:
        L("  nearest named export <= 0xC87A4: %s at +0x%X" % (below_named[1], CLAIM_OFFSET - below_named[0]))
    if below_any:
        L("  nearest any export <= 0xC87A4: rva=0x%X at +0x%X" % (below_any, CLAIM_OFFSET - below_any))
    L("  ASSUMPTION: offset is module-relative (standard WER convention); internal, non-exported code may be closer.")
    L("")

    L("## 7. Windows Application log: python 0xc000070a crashes (last 5 days, read-only PowerShell)")
    out, err, rc = powershell(PS_EVENTS)
    if rc != 0:
        L("  PowerShell rc=%d stderr=%s" % (rc, err[:400]))
    app_rows, wer_rows, lowvm_rows = [], [], []
    for line in out.splitlines():
        if line.startswith("APP|"):
            app_rows.append(line.split("|"))
        elif line.startswith("WER|"):
            wer_rows.append(line.split("|"))
        elif line.startswith("LOWVM|"):
            lowvm_rows.append(line)
    L("  events 0xc000070a app error (Id 1000): %d" % len(app_rows))
    by_path = {}
    for r in app_rows:
        by_path.setdefault(r[2], []).append(r)
    for p, rows in sorted(by_path.items(), key=lambda kv: -len(kv[1])):
        L("    %3d  %s" % (len(rows), p))
    offs = sorted(set(r[5] for r in app_rows))
    L("  distinct fault offsets across all events: %s" % offs)
    lifetimes = []
    for r in app_rows:
        t_crash = dt.datetime.strptime(r[1], "%Y-%m-%d %H:%M:%S")
        t_start = dt.datetime.strptime(r[1][:10] + " " + r[6], "%Y-%m-%d %H:%M:%S")
        life = (t_crash - t_start).total_seconds()
        if life < 0:
            life += 86400
        lifetimes.append((life, "cs2gpu" if "cs2gpu" in r[2] else "G", r[1]))
    lifetimes.sort()
    secs = [x[0] for x in lifetimes]
    L("  process lifetime before crash (s, approx from HH:mm:ss start): min=%.0f median=%.0f max=%.0f ; <10 s: %d of %d" % (
        secs[0], secs[len(secs) // 2], secs[-1], sum(1 for s in secs if s < 10), len(secs)))
    L("  shortest lifetimes: " + "; ".join("%s %s (%.0f s)" % (lab, ts, s) for s, lab, ts in lifetimes[:6]))
    L("  cs2gpu-env crashes: " + "; ".join("%s (%.0f s)" % (r[1], (dt.datetime.strptime(r[1], "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(r[1][:10] + " " + r[6], "%Y-%m-%d %H:%M:%S")).total_seconds()) for r in app_rows if "cs2gpu" in r[2]))
    ts_by_binary = {}
    for r in app_rows:
        ts_by_binary.setdefault(r[4], set()).add("cs2gpu" if "cs2gpu" in r[2] else ("G-project-env" if r[2].startswith("G:") else r[2]))
    L("  binary timestamp -> env label: %s" % {k: sorted(v) for k, v in ts_by_binary.items()})
    L("  events in 16:30-18:00 window (time | path | faulting PID | start (local)):")
    for r in sorted(app_rows, key=lambda x: x[1]):
        if "2026-10-13 16:30" <= r[1] <= "2026-10-13 18:00":
            L("    %s | %s | %s | start %s" % (r[1], "cs2gpu" if "cs2gpu" in r[2] else "G", r[3], r[6]))
    L("  WER Id 1001 entries 17:05-17:09 (P1 app, P3 binary timestamp, P7 code, P8 offset):")
    for r in wer_rows:
        L("    %s | %s | P3=%s | P7=%s | P8=%s" % (r[1], r[2], r[3], r[4], r[5]))
    L("  WER report archive entries modified 17:05-17:09 (read-only):")
    for line in wer_reports(dt.datetime(2026, 10, 13, 17, 5), dt.datetime(2026, 10, 13, 17, 9)):
        L("    " + line)
    L("")

    L("## 8. Low virtual memory detector (System Id 2004), last 5 days")
    total_lowvm = 0
    per_hour = {}
    for line in lowvm_rows:
        total_lowvm += 1
        ts = line.split("|")[1]
        per_hour[ts[:13]] = per_hour.get(ts[:13], 0) + 1
    L("  total low-VM events: %d" % total_lowvm)
    for h in sorted(per_hour):
        L("    %s:00  low-VM events=%d" % (h, per_hour[h]))
    for line in lowvm_rows:
        if "2026-10-13 16:" in line or "2026-10-13 17:" in line:
            tops = re.findall(r"([\w.]+\.exe) \((\d+)\) consumed (\d+) bytes", line)
            ts = line.split("|")[1]
            L("    %s top consumers: %s" % (ts, ", ".join("%s(pid %s)=%.2f GB" % (n, p, int(b) / GB) for n, p, b in tops)))
    L("")

    L("## 9. Current system commit (GetPerformanceInfo) and arithmetic against the claim")
    cur = system_commit()
    L("  now (GiB, 2^30 bytes): commit=%.2f  limit=%.2f  headroom=%.2f  commit_peak_since_boot=%.2f  avail_RAM=%.2f  total_RAM=%.2f" % (
        cur["commit_gb"], cur["limit_gb"], cur["limit_gb"] - cur["commit_gb"], cur["peak_gb"],
        cur["avail_ram_gb"], cur["total_ram_gb"]))
    claim_headroom = 81.2 - 70.7
    L("  claimed snapshot (units not stated in E1.md): commit=70.7 limit=81.2 => headroom=%.2f ; no raw artifact found for these numbers" % claim_headroom)
    L("  claimed free RAM 6.3-7.5 (units not stated; only E1.md).")
    L("")

    L("## 10. Was anything changed in the OS or the Python builds just before the crash series? (read-only)")
    ntdll_st = os.stat(NTDLL)
    L("  on-disk ntdll.dll mtime=%s ctime=%s" % (dt.datetime.fromtimestamp(ntdll_st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                                              dt.datetime.fromtimestamp(ntdll_st.st_ctime).strftime("%Y-%m-%d %H:%M")))
    ps_upd = r"""
$s = [DateTime]'2026-10-10 00:00:00'
$u = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WindowsUpdateClient'; StartTime=$s} -ErrorAction SilentlyContinue
foreach ($e in $u) { Write-Output ('UPD|' + $e.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss') + '|' + $e.Id + '|' + (($e.Message -split '\r?\n')[0])) }
Get-HotFix -ErrorAction SilentlyContinue | Where-Object { $_.InstalledOn -ge [DateTime]'2026-10-08' } | ForEach-Object { Write-Output ('HOTFIX|' + $_.InstalledOn.ToString('yyyy-MM-dd') + '|' + $_.HotFixID) }
"""
    out2, err2, rc2 = powershell(ps_upd)
    upd = [l for l in out2.splitlines() if l.startswith("UPD|")][:20]
    hot = [l for l in out2.splitlines() if l.startswith("HOTFIX|")][:20]
    L("  WindowsUpdateClient events since 10-10 (first 20): %d shown" % len(upd))
    for l in upd:
        L("    " + l[:150])
    L("  hotfixes installed since 10-08: %s" % (hot if hot else "none listed"))
    if rc2 != 0:
        L("  PowerShell stderr (update query): " + err2[:200])
    L("")
    L("STATIC CHECKS DONE. Next: verify_E1_5_repro.py (controls and a 1M-step repeat, appended to this log).")
    L.f.close()


if __name__ == "__main__":
    main()
