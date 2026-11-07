r"""verify E1-5 helper: which OpenMP / MKL runtimes were loaded in each readable python.exe crash (2026-10-13).

Tests the alternative 'two OpenMP runtimes in one process' (e.g. VCOMP140.DLL from xgboost plus libiomp5md.dll
from MKL/sklearn). Read-only: parses Report.wer LoadedModule lists.
Run:
  "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_E1_5_omp.py
"""
import datetime
import glob
import os
import re

ARCHIVE = r"C:\ProgramData\Microsoft\Windows\WER\ReportArchive"
OMP_NAMES = ("vcomp140.dll", "libiomp5md.dll", "libomp.dll", "libomp140.dll", "vcomp.dll", "libgomp", "mkl_rt", "tbb12.dll", "tbb.dll")
rows = []
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
    mods = [os.path.basename(x.strip()).lower() for x in re.findall(r"LoadedModule\[\d+\]=([^\r\n]+)", txt)]
    omp = sorted({x for x in mods if any(n in x for n in OMP_NAMES)})
    rows.append((t, code.group(1) if code else "?", len(mods), omp))

print(f"readable python.exe WER reports on 2026-10-13: {len(rows)}")
for t, code, n, omp in sorted(rows, key=lambda r: r[0]):
    print(f"  {t.strftime('%H:%M:%S')} code={code} modules={n} openmp/mkl runtimes={omp}")
multi = [r for r in rows if len([x for x in r[3] if 'mkl' not in x]) >= 2 or ('vcomp140.dll' in r[3] and any('iomp' in x or 'libomp' in x for x in r[3]))]
print(f"reports with two or more distinct OpenMP runtimes loaded: {len(multi)} of {len(rows)}")
