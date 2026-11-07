# -*- coding: utf-8 -*-
"""verify_E3_3_recount.py

Independent recount for finding E3-3 (claim: a detached C: pixi env costs about
0.026 GiB of unique disk because its files are hard-linked into the rattler cache).

Read-only: directory listings, stat calls, FindFirstFileNameW (hard-link names),
and CreateFileW with FILE_READ_ATTRIBUTES only (no data read, no write). The only
file written is logs\\swarm2\\verify_E3_3.log.

Method
  1. Walk the claimed C: env, the current G: project env, and the three other
     C: pixi envs in the workspace-envs folder.
  2. For every regular file in the claimed C: env, enumerate ALL hard-link names
     with FindFirstFileNameW and classify each name as: the claimed env, the
     rattler cache, another workspace-envs env, or elsewhere. Cross-check the
     enumerated count against st_nlink and FILE_STANDARD_INFO.NumberOfLinks.
  3. On-disk bytes per inode from FILE_STANDARD_INFO.AllocationSize (cluster
     rounded). GetCompressedFileSizeW is NOT used: for uncompressed files it
     returns the logical size.
  4. Four removal scenarios, summed over inodes:
       A  delete the claimed env only (the claim's 'unique' number)
       B  rattler cache copy removed, everything else kept
       C  rattler cache and all other workspace envs removed, elsewhere-links kept
       D  whole tree (upper bound; equals the env's footprint)
  5. Package-set comparison claimed C: vs G: (conda-meta records, relpath+size),
     CUDA-looking files, __pycache__ counts, G:-only sitecustomize.py.
  6. Pixi config files that decide which env 'pixi run' uses; free space now.
"""
import collections
import ctypes
import json
import os
import re
import shutil
import stat
import sys
import time
import traceback
from ctypes import wintypes

LOG = r"G:\pureskill\csds-cs2\logs\swarm2\verify_E3_3.log"
WS = r"C:\Users\13144\.pixi\workspace-envs"
CACHE = r"C:\Users\13144\AppData\Local\rattler\cache"
CLAIM = WS + r"\csds-cs2-15611918389529754563\envs\default"
G_ENV = r"G:\pureskill\csds-cs2\.pixi\envs\default"
OTHER_C = [
    ("cs2gpu", WS + r"\cs2gpu-15282324412032665829\envs\default"),
    ("cs2gpu_alt", WS + r"\cs2gpu_alt-16300978804608365045\envs\default"),
    ("cs2gpu_torch", WS + r"\cs2gpu_torch-7064157186003075779\envs\default"),
]
CONFIGS = [
    r"G:\pureskill\csds-cs2\.pixi\config.toml",
    r"C:\Users\13144\.pixi\config.toml",
    r"G:\pureskill\csds-cs2\.pixi\envs\README.txt",
]
REPARSE = 0x400
GIB = float(1 << 30)
CUDA_RE = re.compile(r"(cudart|cublas|nvrtc|cudnn|cufft|cusparse|nvjitlink|curand|cusolver)", re.I)
PKG_RE = re.compile(r"(cuda|xgboost|lightgbm|catboost|torch|cupy|pytorch)", re.I)

LINES = []


def out(*parts):
    s = " ".join(str(p) for p in parts)
    LINES.append(s)
    print(s, flush=True)


def gib(b):
    return "%.4f GiB (%d B)" % (b / GIB, b)


def lp(p):
    p = os.path.normpath(p)
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
HANDLE = ctypes.c_void_p
INVALID = ctypes.c_void_p(-1).value
ERROR_HANDLE_EOF = 38

kernel32.FindFirstFileNameW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR]
kernel32.FindFirstFileNameW.restype = HANDLE
kernel32.FindNextFileNameW.argtypes = [HANDLE, ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR]
kernel32.FindNextFileNameW.restype = wintypes.BOOL
kernel32.FindClose.argtypes = [HANDLE]
kernel32.FindClose.restype = wintypes.BOOL
kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                 ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
kernel32.CreateFileW.restype = HANDLE
kernel32.CloseHandle.argtypes = [HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.GetFileInformationByHandleEx.argtypes = [HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL


class FILE_STANDARD_INFO(ctypes.Structure):
    _fields_ = [("AllocationSize", ctypes.c_longlong),
                ("EndOfFile", ctypes.c_longlong),
                ("NumberOfLinks", wintypes.DWORD),
                ("DeletePending", wintypes.BOOLEAN),
                ("Directory", wintypes.BOOLEAN)]


def link_names(path):
    """All hard-link names of one file, as drive-qualified paths."""
    cap = 32768
    size = wintypes.DWORD(cap)
    buf = ctypes.create_unicode_buffer(cap)
    h = kernel32.FindFirstFileNameW(lp(path), 0, ctypes.byref(size), buf)
    if h is None or h == INVALID:
        raise OSError(ctypes.get_last_error(), "FindFirstFileNameW failed: " + path)
    raw = [buf.value]
    try:
        while True:
            size.value = cap
            if not kernel32.FindNextFileNameW(h, ctypes.byref(size), buf):
                err = ctypes.get_last_error()
                if err == ERROR_HANDLE_EOF:
                    break
                raise OSError(err, "FindNextFileNameW failed: " + path)
            raw.append(buf.value)
    finally:
        kernel32.FindClose(h)
    drive = os.path.splitdrive(path)[0]
    full = []
    for nm in raw:
        if nm.startswith("\\\\?\\"):
            nm = nm[4:]
        if len(nm) > 1 and nm[1] == ":":
            full.append(nm)
        else:
            full.append(drive + nm)
    return full


def std_info(path):
    h = kernel32.CreateFileW(lp(path), 0x80, 0x7, None, 3, 0x02000000, None)
    if h is None or h == INVALID:
        raise OSError(ctypes.get_last_error(), "CreateFileW failed: " + path)
    try:
        info = FILE_STANDARD_INFO()
        if not kernel32.GetFileInformationByHandleEx(h, 1, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "GetFileInformationByHandleEx failed: " + path)
        return info.AllocationSize, info.NumberOfLinks
    finally:
        kernel32.CloseHandle(h)


def walk(root):
    recs = []
    reparse = []
    errs = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(lp(d)) as it:
                entries = list(it)
        except OSError as e:
            errs.append((d, repr(e)))
            continue
        for ent in entries:
            full = os.path.join(d, ent.name)
            try:
                st = os.stat(lp(full), follow_symlinks=False)
            except OSError as e:
                errs.append((full, repr(e)))
                continue
            if getattr(st, "st_file_attributes", 0) & REPARSE:
                reparse.append(full)
                continue
            if stat.S_ISDIR(st.st_mode):
                stack.append(full)
            elif stat.S_ISREG(st.st_mode):
                recs.append((full, os.path.relpath(full, root), st.st_size,
                             st.st_nlink, st.st_dev, st.st_ino))
    return recs, reparse, errs


CLAIM_L = CLAIM.lower()
CACHE_L = CACHE.lower()
WS_L = WS.lower()


def bucket(full):
    f = full.lower()
    if f == CLAIM_L or f.startswith(CLAIM_L + "\\"):
        return ("claim", "")
    if f.startswith(CACHE_L + "\\"):
        return ("cache", "")
    if f.startswith(WS_L + "\\"):
        return ("ws", full[len(WS) + 1:].split("\\")[0])
    return ("oth", "\\".join(full.split("\\")[:4]))


def conda_records(root):
    d = os.path.join(root, "conda-meta")
    recs = {}
    for fn in sorted(os.listdir(lp(d))):
        if not fn.endswith(".json"):
            continue
        with open(lp(os.path.join(d, fn)), encoding="utf-8") as fh:
            j = json.load(fh)
        recs[fn] = (j.get("name"), j.get("version"), j.get("build"),
                    j.get("sha256") or j.get("md5"))
    return recs


def pct_row(label, counts):
    return "%s" % (label,)


def main():
    t0 = time.time()
    out("verify_E3_3_recount.py started", time.strftime("%Y-%m-%d %H:%M:%S"))
    out("interpreter:", sys.executable, "version", sys.version.split()[0])
    out("GiB = 2**30 bytes. logical = st_size. on-disk = FILE_STANDARD_INFO.AllocationSize.")
    out("")

    for drv in ("C:\\", "G:\\"):
        du = shutil.disk_usage(drv)
        out("free now %s %.2f GiB of %.2f GiB" % (drv, du.free / GIB, du.total / GIB))
    out("")

    out("== pixi config files (decide which env 'pixi run' uses for this project)")
    for p in CONFIGS:
        try:
            st = os.stat(lp(p))
            out("file %s mtime=%s size=%d" % (p, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)), st.st_size))
            with open(lp(p), encoding="utf-8", errors="replace") as fh:
                for line in fh.read().splitlines()[:6]:
                    out("    | " + line)
        except OSError as e:
            out("file %s read failed: %r" % (p, e))
    out("")

    envs = collections.OrderedDict()
    labels = [("claim", CLAIM), ("G", G_ENV)] + [(lab, root) for lab, root in OTHER_C]
    out("== walks (regular files, bytes, __pycache__ files)")
    for label, root in labels:
        t1 = time.time()
        recs, reparse, errs = walk(root)
        envs[label] = dict(root=root, recs=recs, reparse=reparse, errs=errs)
        apparent = sum(r[2] for r in recs)
        pyc = [r for r in recs if "__pycache__" in r[1]]
        out("%-12s files=%6d apparent=%s reparse=%d walk_errors=%d pycache_files=%d pycache_bytes=%d (%.1fs)"
            % (label, len(recs), gib(apparent), len(reparse), len(errs), len(pyc),
               sum(r[2] for r in pyc), time.time() - t1))
    out("")

    # Enumerate every hard-link name of every file in the claimed env.
    claim_recs = envs["claim"]["recs"]
    claim_paths_by_key = collections.Counter((r[4], r[5]) for r in claim_recs)
    inodes = []
    mism_enum = mism_info = enum_fail = claim_not_one = 0
    t1 = time.time()
    seen = set()
    for full, rel, size, nlink, dev, ino in claim_recs:
        key = (dev, ino)
        if key in seen:
            continue
        seen.add(key)
        try:
            names = link_names(full)
        except OSError as e:
            enum_fail += 1
            names = [full]
            out("enum fail %s: %r" % (rel, e))
        if len(names) != nlink:
            mism_enum += 1
        try:
            alloc, ninfo = std_info(full)
        except OSError as e:
            alloc, ninfo = size, -1
            out("alloc fail %s: %r" % (rel, e))
        if ninfo != nlink:
            mism_info += 1
        n_claim = n_cache = n_ws = n_oth = 0
        ws_set = set()
        oth_set = set()
        for nm in names:
            b, sub = bucket(nm)
            if b == "claim":
                n_claim += 1
            elif b == "cache":
                n_cache += 1
            elif b == "ws":
                n_ws += 1
                ws_set.add(sub)
            else:
                n_oth += 1
                oth_set.add(sub)
        if n_claim != claim_paths_by_key[key]:
            claim_not_one += 1
        inodes.append(dict(rel=rel, logical=size, alloc=alloc, nlink=nlink,
                           n_claim=n_claim, n_cache=n_cache, n_ws=n_ws, n_oth=n_oth,
                           ws=ws_set, oth=oth_set, names=names))
    out("claimed env: enumerated hard-link names for %d inodes in %.1fs"
        % (len(inodes), time.time() - t1))
    out("  enumerated-name count != st_nlink: %d ; FILE_STANDARD_INFO.NumberOfLinks != st_nlink: %d ; enumeration failures: %d"
        % (mism_enum, mism_info, enum_fail))
    out("  claim-env name count != walked path count: %d" % claim_not_one)
    out("")

    total_logical = sum(i["logical"] for i in inodes)
    total_alloc = sum(i["alloc"] for i in inodes)
    out("claimed env totals: logical=%s  on-disk=%s" % (gib(total_logical), gib(total_alloc)))
    n_cache_inodes = sum(1 for i in inodes if i["n_cache"] > 0)
    out("  inodes with >=1 rattler-cache name: %d of %d" % (n_cache_inodes, len(inodes)))
    out("  inodes with nlink==1: %d" % sum(1 for i in inodes if i["nlink"] == 1))
    out("  inodes with another workspace-envs name: %d" % sum(1 for i in inodes if i["n_ws"] > 0))
    out("  inodes with an 'elsewhere' name: %d" % sum(1 for i in inodes if i["n_oth"] > 0))
    out("")

    out("== removal scenarios for the claimed env (sums over inodes)")
    scen = [
        ("A delete claimed env only (claim's 'unique now')",
         lambda i: i["n_cache"] == 0 and i["n_ws"] == 0 and i["n_oth"] == 0),
        ("B rattler cache copy removed, others kept",
         lambda i: i["n_ws"] == 0 and i["n_oth"] == 0),
        ("C rattler cache + other workspace envs removed, elsewhere kept",
         lambda i: i["n_oth"] == 0),
        ("D whole tree (upper bound)", lambda i: True),
    ]
    for title, pred in scen:
        sel = [i for i in inodes if pred(i)]
        out("  %-62s inodes=%-6d logical=%s on-disk=%s"
            % (title, len(sel), gib(sum(i["logical"] for i in sel)), gib(sum(i["alloc"] for i in sel))))
    out("")

    pat = collections.Counter()
    for i in inodes:
        pat[(min(i["n_cache"], 1), min(i["n_ws"], 9), min(i["n_oth"], 9))] += 1
    out("pattern (cache-linked 0/1, other-ws names, elsewhere names) -> inode count")
    for k, v in sorted(pat.items(), key=lambda kv: -kv[1])[:12]:
        out("  %s -> %d" % (k, v))
    out("")

    ws_cnt = collections.Counter()
    for i in inodes:
        for s in i["ws"]:
            ws_cnt[s] += 1
    out("other workspace-envs folders referenced by claimed-env inodes: %d distinct" % len(ws_cnt))
    for s, v in ws_cnt.most_common(8):
        out("  %6d inodes  %s" % (v, s))
    oth_cnt = collections.Counter()
    for i in inodes:
        for s in i["oth"]:
            oth_cnt[s] += 1
    out("elsewhere locations referenced by claimed-env inodes: %d distinct prefixes" % len(oth_cnt))
    for s, v in oth_cnt.most_common(8):
        out("  %6d inodes  %s" % (v, s))
    out("")

    # Unique-now list (what the number is made of).
    uniq = sorted((i for i in inodes if i["n_cache"] == 0 and i["n_ws"] == 0 and i["n_oth"] == 0),
                  key=lambda i: -i["logical"])
    out("largest files freed by deleting the claimed env alone (scenario A):")
    for i in uniq[:10]:
        out("  %10d B  alloc %10d B  %s" % (i["logical"], i["alloc"], i["rel"]))
    out("")

    # Package-set comparison: claimed C: env vs G: project env.
    out("== package-set comparison: claimed C: env vs G: project env")
    cmc = conda_records(CLAIM)
    cmg = conda_records(G_ENV)
    out("conda-meta records: C=%d G=%d" % (len(cmc), len(cmg)))
    only_c = sorted(set(cmc) - set(cmg))
    only_g = sorted(set(cmg) - set(cmc))
    diff = sorted(k for k in set(cmc) & set(cmg) if cmc[k] != cmg[k])
    out("  only in C: %s" % only_c)
    out("  only in G: %s" % only_g)
    out("  shared records with different name/version/build/sha: %d %s" % (len(diff), diff[:5]))
    for lab, cm in (("C", cmc), ("G", cmg)):
        hits = sorted(set((v[0], v[1], v[2]) for v in cm.values() if v[0] and PKG_RE.search(v[0])))
        out("  %s GPU/ML-relevant records: %s" % (lab, hits))
    out("")

    rc = {r[1]: r[2] for r in envs["claim"]["recs"]}
    rg = {r[1]: r[2] for r in envs["G"]["recs"]}
    common = set(rc) & set(rg)
    size_diff = [k for k in common if rc[k] != rg[k]]
    only_rc = sorted(set(rc) - set(rg))
    only_rg = sorted(set(rg) - set(rc))
    out("relpath manifest: C=%d G=%d common=%d same-relpath-different-size=%d"
        % (len(rc), len(rg), len(common), len(size_diff)))
    out("  only in C (%d): %s" % (len(only_rc), only_rc[:10]))
    out("  only in G (%d): %s" % (len(only_rg), [p for p in only_rg[:6]]))
    only_rg_pyc = [p for p in only_rg if "__pycache__" in p]
    out("  of the only-in-G files, __pycache__ .pyc: %d files, %d bytes"
        % (len(only_rg_pyc), sum(rg[p] for p in only_rg_pyc)))
    other_only_rg = [p for p in only_rg if "__pycache__" not in p]
    out("  of the only-in-G files, not pycache: %s" % other_only_rg)
    out("")

    for lab in ("claim", "G", "cs2gpu"):
        recs = envs[lab]["recs"]
        names = sorted(set(os.path.basename(r[0]) for r in recs if CUDA_RE.search(os.path.basename(r[0]))))
        out("CUDA-looking file names in %-8s : %d distinct %s" % (lab, len(names), names[:10]))
    out("")

    sc = G_ENV + r"\Lib\site-packages\sitecustomize.py"
    try:
        st = os.stat(lp(sc))
        with open(lp(sc), encoding="utf-8", errors="replace") as fh:
            head = fh.read(400)
        out("G-only file sitecustomize.py: size=%d mtime=%s" % (st.st_size, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))))
        out("  first 400 chars: " + repr(head))
    except OSError as e:
        out("sitecustomize.py not readable in G env: %r" % (e,))
    out("")

    out("total runtime %.1fs" % (time.time() - t0))
    out("finished", time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        out("EXCEPTION:")
        out(traceback.format_exc())
    finally:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(LINES) + "\n")
