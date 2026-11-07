"""verify E1-5: Windows event-log and WER evidence for the 0xC000070A native crashes.

Tests the claims in notes/swarm2/E1.md (E1-5):
  (a) the four cited APPCRASH times (17:06:16, 17:06:56, 17:07:34, 17:07:38)
      and the claim that all four are this env's python.exe;
  (b) the E1_parity process died in its 1M step (process start time, crash time,
      parity log write time);
  (c) whether the crash is specific to the 1M workload or appears across processes
      and days (exception code 0xC000070A, ntdll offset 0xC87A4);
  (d) whether low-virtual-memory events (Resource-Exhaustion-Detector 2004) line up
      with crash bursts (commit-pressure hypothesis);
  (e) which non-standard DLLs were loaded in the crashed processes (injection or
      security-hook alternative), read from WER Report.wer LoadedModule lists.
Read-only: uses wevtutil query output and reads Report.wer files under ReportArchive.

Run:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_E1_5_events.py
"""
import datetime
import glob
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"
ARCHIVE = r"C:\ProgramData\Microsoft\Windows\WER\ReportArchive"
PARITY_SCRIPT = r"G:\pureskill\csds-cs2\scripts\swarm2\E1_parity.py"
PARITY_LOG = r"G:\pureskill\csds-cs2\logs\swarm2\E1_parity.log"
BUILD_G = "6ac586fc"  # time stamp of G: project env python.exe (from 1000/1001 events)
BUILD_CS2GPU = "6ace601b"  # time stamp of cs2gpu env python.exe


def local_from_utc_iso(s):
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = head + "." + frac[:6]
    dt = datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone()


def wevt(log, xpath):
    r = subprocess.run(["wevtutil", "qe", log, "/q:" + xpath, "/f:xml", "/rd:true"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    body = r.stdout or ""
    if not body.strip():
        return []
    root = ET.fromstring("<Events>" + body + "</Events>")
    out = []
    for ev in root.findall(NS + "Event"):
        sysn = ev.find(NS + "System")
        eid = int(sysn.find(NS + "EventID").text)
        prov = sysn.find(NS + "Provider").get("Name")
        t = local_from_utc_iso(sysn.find(NS + "TimeCreated").get("SystemTime"))
        data = {}
        seq = []
        ed = ev.find(NS + "EventData")
        if ed is not None:
            for i, d in enumerate(ed.findall(NS + "Data")):
                name = d.get("Name")
                txt = d.text or ""
                if name:
                    data[name] = txt
                seq.append(txt)
        msg = ev.find(NS + "RenderingInfo")
        out.append({"id": eid, "prov": prov, "t": t, "data": data, "seq": seq})
    return out


def fmt(t):
    return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def text_messages(log, xpath):
    """Rendered text per event, keyed by local HH:MM:SS of the event."""
    r = subprocess.run(["wevtutil", "qe", log, "/q:" + xpath, "/f:text", "/rd:true"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = {}
    for chunk in (r.stdout or "").split("Event[")[1:]:
        m = re.search(r"Date:\s+(\S+)T(\d\d:\d\d:\d\d)", chunk)
        if not m:
            continue
        out.setdefault(m.group(2), chunk)
    return out


def crash_fields(e):
    """Map a 1000 event to named fields. Falls back to positional EventData order."""
    d = e["data"]
    seq = e["seq"]
    f = {}
    f["app"] = d.get("AppName") or (seq[0] if len(seq) > 0 else "")
    f["appts"] = d.get("AppTimeStamp") or (seq[2] if len(seq) > 2 else "")
    f["mod"] = d.get("ModuleName") or (seq[3] if len(seq) > 3 else "")
    f["code"] = (d.get("ExceptionCode") or (seq[6] if len(seq) > 6 else "")).lower()
    f["offset"] = (d.get("FaultingOffset") or (seq[7] if len(seq) > 7 else "")).lower()
    f["pid"] = d.get("ProcessId") or (seq[8] if len(seq) > 8 else "")
    f["start"] = d.get("ProcessCreationTime") or (seq[9] if len(seq) > 9 else "")
    f["path"] = d.get("AppPath") or (seq[10] if len(seq) > 10 else "")
    return f


def main():
    print("=== verify_E1_5_events.py ===")
    print("now local:", datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))

    # ---- load Application Error (1000) and WER (1001) events since 2026-10-12 UTC ----
    apps = wevt("Application",
                "*[System[(EventID=1000 or EventID=1001) and TimeCreated[@SystemTime>='2026-10-11T00:00:00.000Z']]]")
    ae = [e for e in apps if e["id"] == 1000]
    wer = [e for e in apps if e["id"] == 1001]
    print(f"loaded since 2026-10-11 (UTC): 1000 events={len(ae)}, 1001 events={len(wer)}")
    if ae:
        print("1000 EventData names (first event):", sorted(ae[0]["data"].keys()) or "(positional only)")
        print("1000 positional sample:", [x[:60] for x in ae[0]["seq"][:12]])

    # ---- (c) crash inventory by day, app, code ----
    print("\n[c] 1000 events by day / faulting app / exception code")
    inv = Counter()
    for e in ae:
        f = crash_fields(e)
        inv[(e["t"].strftime("%Y-%m-%d"), os.path.basename(f["app"]), f["code"])] += 1
    for k, v in sorted(inv.items()):
        print(f"  {k[0]}  {k[1]:<14} {k[2]:<12} n={v}")

    crash_all = [e for e in ae if crash_fields(e)["code"] == "c000070a"]
    print(f"\n[c] 0xc000070a total (all days in load): {len(crash_all)}")
    print("[c] first 0xc000070a event:", fmt(min(e["t"] for e in crash_all)) if crash_all else "none")
    print("[c] last  0xc000070a event:", fmt(max(e["t"] for e in crash_all)) if crash_all else "none")
    ov = Counter(crash_fields(e)["offset"] for e in crash_all)
    print("[c] fault offsets:", dict(ov))
    mods = Counter(crash_fields(e)["mod"] for e in crash_all)
    print("[c] faulting modules:", dict(mods))
    builds = Counter(crash_fields(e)["appts"] for e in crash_all)
    print("[c] app build stamps:", dict(builds))
    paths_g = sum(1 for e in crash_all if crash_fields(e)["appts"] == BUILD_G)
    paths_c = sum(1 for e in crash_all if crash_fields(e)["appts"] == BUILD_CS2GPU)
    print(f"[c] by build: G: env (stamp {BUILD_G}) n={paths_g}; cs2gpu env (stamp {BUILD_CS2GPU}) n={paths_c}")

    hourly = Counter(e["t"].strftime("%m-%d %H:00") for e in crash_all)
    print("[c] 0xc000070a per hour (local):")
    for k in sorted(hourly):
        print(f"  {k}  n={hourly[k]}")

    # ---- (a) the four cited times ----
    wer_seq_shown = 0
    for e in wer:
        if e["t"].strftime("%Y-%m-%d") == "2026-10-13" and e["t"].strftime("%H:%M") in ("17:06", "17:07") and wer_seq_shown < 4:
            vals = [x for x in e["seq"] if x][:12]
            print("  1001 positional fields at", fmt(e["t"]), [x[:50] for x in vals])
            wer_seq_shown += 1
    print("\n[a] cited APPCRASH times: nearest 1000/1001 events within +/-3 s")
    cited = ["17:06:16", "17:06:56", "17:07:34", "17:07:38"]
    day = "2026-10-13"
    tz = datetime.datetime.now().astimezone().tzinfo
    for hms in cited:
        target = datetime.datetime.fromisoformat(f"{day}T{hms}").replace(tzinfo=tz)
        near = [e for e in apps if abs((e["t"] - target).total_seconds()) <= 3]
        if not near:
            print(f"  {hms}: NO 1000/1001 event within 3 s")
            continue
        for e in near:
            if e["id"] == 1000:
                f = crash_fields(e)
                build = "G:env" if f["appts"] == BUILD_G else ("cs2gpu" if f["appts"] == BUILD_CS2GPU else f["appts"])
                print(f"  {hms}: {fmt(e['t'])} id=1000 {f['app']} code={f['code']} off={f['offset']} "
                      f"build={build} pid={f['pid']} path={f['path']}")
            else:
                seq = e["seq"]
                build_1001 = seq[seq.index("python.exe") + 2] if "python.exe" in seq and len(seq) > seq.index("python.exe") + 2 else "?"
                b = "G:env" if build_1001 == BUILD_G else ("cs2gpu" if build_1001 == BUILD_CS2GPU else build_1001)
                exc_1001 = seq[seq.index("ntdll.dll") + 3] if "ntdll.dll" in seq and len(seq) > seq.index("ntdll.dll") + 3 else "?"
                print(f"  {hms}: {fmt(e['t'])} id=1001 WER APPCRASH python.exe build={b} exc={exc_1001} (no PID field in 1001)")

    # ---- (b) E1_parity process identity and timing ----
    print("\n[b] E1_parity process identity")
    cs2 = [e for e in ae if crash_fields(e)["appts"] == BUILD_CS2GPU and crash_fields(e)["code"] == "c000070a"]
    for e in sorted(cs2, key=lambda x: x["t"]):
        f = crash_fields(e)
        try:
            ft = int(f["start"], 16) if f["start"] else 0
        except ValueError:
            ft = 0
        if ft:
            start_utc = datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(microseconds=ft / 10)
            start_local = start_utc.astimezone()
        else:
            start_local = None
        try:
            pid_int = int(f["pid"], 16) if str(f["pid"]).lower().startswith("0x") else int(f["pid"] or 0)
        except ValueError:
            pid_int = -1
        print(f"  crash {fmt(e['t'])} pid={pid_int} (0x{pid_int:X}) start={fmt(start_local) if start_local else 'n/a'} path={f['path']}")
    if os.path.exists(PARITY_SCRIPT):
        print("  E1_parity.py mtime  ", datetime.datetime.fromtimestamp(os.path.getmtime(PARITY_SCRIPT)).astimezone().strftime("%H:%M:%S.%f")[:-3])
    if os.path.exists(PARITY_LOG):
        print("  E1_parity.log mtime ", datetime.datetime.fromtimestamp(os.path.getmtime(PARITY_LOG)).astimezone().strftime("%H:%M:%S.%f")[:-3],
              "(last write = 100k summary line; no 1M line in the log)")
    with open(PARITY_LOG, encoding="utf-8-sig") as fh:
        lines = [x.rstrip() for x in fh if x.strip()]
    print("  E1_parity.log lines:")
    for x in lines:
        print("    " + x)
    print("  1M lines in E1_parity.log:", sum(1 for x in lines if "1,000,000" in x or "1000000" in x))

    # ---- (d) low virtual memory events vs crash bursts ----
    print("\n[d] Resource-Exhaustion-Detector 2004 (low virtual memory), 2026-10-12 onward")
    sysev = wevt("System", "*[System[(EventID=2004) and TimeCreated[@SystemTime>='2026-10-12T00:00:00.000Z']]]")
    sysev = [e for e in sysev if e["prov"] == "Microsoft-Windows-Resource-Exhaustion-Detector"]
    msgs = text_messages("System", "*[System[(EventID=2004) and TimeCreated[@SystemTime>='2026-10-12T00:00:00.000Z']]]")
    print(f"  2004 events: {len(sysev)} (messages parsed: {len(msgs)})")
    crash_times = sorted(e["t"] for e in crash_all)
    w0 = datetime.datetime.fromisoformat("2026-10-13T16:00:00").replace(tzinfo=tz)
    w1 = datetime.datetime.now().astimezone()
    win = [t for t in crash_times if w0 <= t < w1]
    print(f"  0xc000070a crashes in 16:00-now local: {len(win)}")
    nbin = int((w1 - w0).total_seconds() // 600) + 1
    bins = [0] * nbin
    for t in win:
        bins[int((t - w0).total_seconds() // 600)] += 1
    print("  10-min bins with >=1 crash in 16:00-now: %d of %d" % (sum(1 for b in bins if b > 0), nbin))
    print("  2004 events in 16:00-now, with crashes in following 10 min (count) and top consumers:")
    for e in sorted([x for x in sysev if w0 <= x["t"] < w1], key=lambda x: x["t"]):
        after = [t for t in crash_times if e["t"] <= t <= e["t"] + datetime.timedelta(minutes=10)]
        m = msgs.get(e["t"].strftime("%H:%M:%S"), "")
        top = re.findall(r"(python[\d.]*\.exe \(\d+\)) consumed (\d+) bytes", m)
        tops_txt = ", ".join(f"{a}={int(b) / 2**30:.2f}GB" for a, b in top) or m[:120]
        print(f"   {e['t'].strftime('%H:%M:%S')} -> crashes next 10 min: {len(after)} | {tops_txt}")
    print("  2004 events before 16:00 (for context, time only):",
          ", ".join(x["t"].strftime("%m-%d %H:%M") for x in sorted(sysev, key=lambda x: x["t"]) if x["t"] < w0))

    # ---- TDR / display driver resets (GPU driver stall alternative) ----
    tdr = wevt("System", "*[System[(EventID=4101) and TimeCreated[@SystemTime>='2026-10-12T00:00:00.000Z']]]")
    print(f"\n[alt] Display 4101 (driver timeout/reset) since 10-12: {len(tdr)}")
    for e in sorted(tdr, key=lambda x: x["t"])[:10]:
        print("   ", fmt(e["t"]), e["prov"])

    # ---- (f) process lifetime at crash, and overlap with 2004 top consumers ----
    print("\n[f] lifetime from process start to 0xc000070a crash (all 60 events)")
    life = []
    pids = set()
    for e in crash_all:
        f = crash_fields(e)
        try:
            ft = int(f["start"], 16)
        except ValueError:
            continue
        st = (datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(microseconds=ft / 10)).astimezone()
        life.append((e["t"] - st).total_seconds())
        try:
            pids.add(int(f["pid"], 16) if str(f["pid"]).lower().startswith("0x") else int(f["pid"]))
        except ValueError:
            pass
    life.sort()
    if life:
        def q(p):
            return life[min(len(life) - 1, int(p * len(life)))]
        print(f"  n={len(life)} min={life[0]:.1f}s p25={q(0.25):.1f}s median={q(0.5):.1f}s p75={q(0.75):.1f}s max={life[-1]:.1f}s")
        print(f"  crashes within 10 s of process start: {sum(1 for x in life if x <= 10)}; within 60 s: {sum(1 for x in life if x <= 60)}")
    top_pids = set()
    for m in re.finditer(r"python[\d.]*\.exe \((\d+)\) consumed", " ".join(msgs.values())):
        top_pids.add(int(m.group(1)))
    print(f"  PIDs named as top VM consumers in 2004 messages: {sorted(top_pids)}")
    print(f"  overlap with crashed PIDs: {sorted(top_pids & pids)} (empty = the crashed processes were not the named top consumers)")

    # ---- (e) loaded modules in WER reports of python crashes on 10-13 ----
    print("\n[e] WER Report.wer LoadedModule lists for python.exe crashes (2026-10-13)")
    reports = []
    unreadable = 0
    for d in glob.glob(os.path.join(ARCHIVE, "AppCrash_python.exe_*")):
        rp = os.path.join(d, "Report.wer")
        if not os.path.exists(rp):
            continue
        try:
            with open(rp, "rb") as fh:
                raw = fh.read()
        except OSError:
            unreadable += 1
            continue
        try:
            txt = raw.decode("utf-16")
        except UnicodeDecodeError:
            txt = raw.decode("utf-8", errors="replace")
        m = re.search(r"EventTime=(\d+)", txt)
        if not m:
            continue
        ft = int(m.group(1))
        t = (datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(microseconds=ft / 10)).astimezone()
        if t.strftime("%Y-%m-%d") != "2026-10-13":
            continue
        code = re.search(r"Sig\[6\]\.Value=(\S+)", txt)
        off = re.search(r"Sig\[7\]\.Value=(\S+)", txt)
        app_ts = re.search(r"Sig\[2\]\.Value=(\S+)", txt)
        mods_list = [x.strip() for x in re.findall(r"LoadedModule\[\d+\]=([^\r\n]+)", txt)]
        reports.append({"t": t, "code": code.group(1) if code else "", "off": off.group(1) if off else "",
                        "ts": app_ts.group(1) if app_ts else "", "mods": mods_list})
    reports.sort(key=lambda r: r["t"])
    print(f"  python.exe WER reports with Report.wer on 10-13: {len(reports)} (unreadable, access denied: {unreadable})")
    print("  by exception code:", dict(Counter(r["code"] for r in reports)))
    print("  by app build stamp:", dict(Counter(r["ts"] for r in reports)))
    common = Counter()
    for r in reports:
        for m in set(os.path.basename(x).lower() for x in r["mods"]):
            common[m] += 1
    std_dirs = ("c:\\windows\\", "\\.pixi\\envs\\", "\\workspace-envs\\", "c:\\program files\\", "c:\\programdata\\")
    nonstd = {}
    for r in reports:
        for x in r["mods"]:
            lx = x.lower()
            if not any(s in lx for s in std_dirs):
                nonstd[x] = nonstd.get(x, 0) + 1
    print("  modules outside WINDOWS / Program Files / python env dirs (count of reports loading it):")
    if nonstd:
        for k, v in sorted(nonstd.items(), key=lambda kv: -kv[1])[:40]:
            print(f"    {v:>3}  {k}")
    else:
        print("    (none)")
    print("  modules present in ALL reports that include a LoadedModule list:")
    n_all = len(reports)
    allmods = [k for k, v in common.items() if v == n_all and n_all > 0]
    print("   ", sorted(allmods)[:60])
    print("  counts of xgboost/omp/cuda-related modules across reports:")
    for key in ("xgboost.dll", "vcomp140.dll", "nvcuda.dll", "libxgboost.dll"):
        print(f"    {key:<16} {common.get(key, 0)}")

    # ---- (a) cross-check parity-process crash counts by env ----
    print("\nsummary")
    print(f"  0xc000070a 1000-events: {len(crash_all)} total; 10-13: {sum(1 for e in crash_all if e['t'].strftime('%Y-%m-%d') == '2026-10-13')}")
    print(f"  G: env build: {paths_g}; cs2gpu build: {paths_c}")
    print(f"  2004 low-VM events since 10-12: {len(sysev)}")


if __name__ == "__main__":
    main()
