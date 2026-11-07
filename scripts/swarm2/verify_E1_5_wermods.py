r"""verify E1-5 helper: list the last loaded modules and the crash point for each readable python.exe WER report (2026-10-13).

Reads Report.wer only (read-only). Shows where in startup each 0xc000070a process was when it faulted:
the module list is in load order, so the tail shows the most recent imports.
Run:
  "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_E1_5_wermods.py
"""
import datetime
import glob
import os
import re

ARCHIVE = r"C:\ProgramData\Microsoft\Windows\WER\ReportArchive"
for d in sorted(glob.glob(os.path.join(ARCHIVE, "AppCrash_python.exe_*"))):
    rp = os.path.join(d, "Report.wer")
    if not os.path.exists(rp):
        continue
    try:
        with open(rp, "rb") as fh:
            raw = fh.read()
    except OSError:
        continue
    txt = raw.decode("utf-16", errors="replace")
    m = re.search(r"EventTime=(\d+)", txt)
    if not m:
        continue
    t = (datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(microseconds=int(m.group(1)) / 10)).astimezone()
    if t.strftime("%Y-%m-%d") != "2026-10-13":
        continue
    code = re.search(r"Sig\[6\]\.Value=(\S+)", txt)
    mods = [x.strip() for x in re.findall(r"LoadedModule\[\d+\]=([^\r\n]+)", txt)]
    print(f"{t.strftime('%H:%M:%S')} code={code.group(1) if code else '?'} modules={len(mods)}")
    for x in mods[-8:]:
        print("    last:", os.path.basename(x))
