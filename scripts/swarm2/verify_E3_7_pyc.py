"""verify_E3_7 pyc/data probe (read-only): does the C: env copy recompile .py on import?

Hypothesis tested: the C: copy has missing or stale .pyc files, so each import recompiles
source. Checks, per copy, for the .py files of the packages used in the timing runs:
  - missing: no __pycache__/<name>.cpython-312.pyc next to the source
  - stale:   pyc header (magic, flags, source mtime, source size) does not match the source
Also sums the parquet bytes under the two G: tome roots (data volume for the pipeline).
Reads only. Writes nothing (sys.dont_write_bytecode set).

Run:
  G:/pureskill/csds-cs2/.pixi/envs/default/python.exe -I -X utf8 G:/pureskill/csds-cs2/scripts/swarm2/verify_E3_7_pyc.py
"""
import os
import struct
import sys
import time

sys.dont_write_bytecode = True

G_ENV = "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default"
C_ENV = "C:\\Users\\<user>\\.pixi\\workspace-envs\\csds-cs2-15611918389529754563\\envs\\default"
SP = os.path.join("Lib", "site-packages")
PKGS = ["numpy", "pandas", "polars", "xgboost", "sklearn", "scipy", "pyarrow", "duckdb"]
TOMES = [
    "G:\\pureskill\\csds-cs2\\tomes\\2026-10-01_to_08\\tome\\csds",
    "G:\\pureskill\\csds-cs2\\tomes\\2024-02-g\\tome\\csds",
]
TAG = "cpython-312"


def out(s=""):
    print(s, flush=True)


def check_copy(root):
    base = os.path.join(root, SP)
    res = {p: {"py": 0, "missing": 0, "stale": 0, "ok": 0} for p in PKGS}
    for pkg in PKGS:
        pdir = os.path.join(base, pkg)
        if not os.path.isdir(pdir):
            continue
        for dp, dns, fns in os.walk(pdir):
            for fn in fns:
                if not fn.endswith(".py"):
                    continue
                src = os.path.join(dp, fn)
                res[pkg]["py"] += 1
                pyc = os.path.join(dp, "__pycache__", fn[:-3] + "." + TAG + ".pyc")
                if not os.path.isfile(pyc):
                    res[pkg]["missing"] += 1
                    continue
                try:
                    st = os.stat(src)
                    with open(pyc, "rb") as fh:
                        hdr = fh.read(16)
                    if len(hdr) < 16:
                        res[pkg]["stale"] += 1
                        continue
                    _magic, flags = struct.unpack("<4sI", hdr[:8])
                    if flags & 1:
                        # hash-based pyc (unchecked or checked): mtime field is not a timestamp.
                        # Not compared here; counted separately so it is not read as stale.
                        res[pkg].setdefault("hash", 0)
                        res[pkg]["hash"] += 1
                        continue
                    mt, sz = struct.unpack("<II", hdr[8:16])
                    if int(st.st_mtime) & 0xFFFFFFFF == mt and st.st_size & 0xFFFFFFFF == sz:
                        res[pkg]["ok"] += 1
                    else:
                        res[pkg]["stale"] += 1
                except OSError:
                    res[pkg]["stale"] += 1
    return res


def section_pyc():
    out("== 1. .pyc coverage for the timing-run packages ==")
    for label, root in [("G", G_ENV), ("C", C_ENV)]:
        t0 = time.perf_counter()
        r = check_copy(root)
        tot = {k: sum(r[p][k] for p in PKGS) for k in ["py", "missing", "stale", "ok"]}
        out("  copy=%s  total py=%d ok=%d missing=%d stale=%d  (%.1fs)"
            % (label, tot["py"], tot["ok"], tot["missing"], tot["stale"], time.perf_counter() - t0))
        for p in PKGS:
            v = r[p]
            if v["py"]:
                out("    %-9s py=%5d ok=%5d missing=%5d stale=%5d hash_based=%5d"
                    % (p, v["py"], v["ok"], v["missing"], v["stale"], v.get("hash", 0)))


def section_tomes():
    out("== 2a. File extension census under the G: tome roots (what the data actually is) ==")
    for root in TOMES:
        ext = {}
        for dp, dns, fns in os.walk(root):
            for fn in fns:
                e = os.path.splitext(fn)[1].lower() or "(none)"
                c = ext.setdefault(e, [0, 0])
                c[0] += 1
                try:
                    c[1] += os.stat(os.path.join(dp, fn)).st_size
                except OSError:
                    pass
        out("  %s" % root)
        for e, (n, b) in sorted(ext.items(), key=lambda kv: -kv[1][1])[:12]:
            out("    ext=%-12s files=%7d  bytes=%10.1f MiB" % (e, n, b / 1048576.0))
    out("== 2. Parquet bytes under the G: tome roots (pipeline data volume) ==")
    for root in TOMES:
        n = 0
        b = 0
        dfn = 0
        dfb = 0
        t0 = time.perf_counter()
        for dp, dns, fns in os.walk(root):
            for fn in fns:
                if fn.endswith(".parquet"):
                    try:
                        s = os.stat(os.path.join(dp, fn)).st_size
                    except OSError:
                        continue
                    n += 1
                    b += s
                    if fn.startswith("dataframe_"):
                        dfn += 1
                        dfb += s
        out("  %s" % root)
        out("    all parquet files=%d  bytes=%.1f MiB  |  dataframe_* files=%d  bytes=%.1f MiB  (walk %.1fs)"
            % (n, b / 1048576.0, dfn, dfb / 1048576.0, time.perf_counter() - t0))


def main():
    out("verify_E3_7_pyc.py  interpreter=%s" % sys.executable)
    section_pyc()
    section_tomes()
    out("== done (read-only) ==")


if __name__ == "__main__":
    main()
