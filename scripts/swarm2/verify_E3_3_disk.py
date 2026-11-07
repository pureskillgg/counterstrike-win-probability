# -*- coding: utf-8 -*-
"""Independent check of finding E3-3 (claim: detached C: env costs ~0.026 GiB unique disk).

Recomputes from the filesystem only:
  * per-env file count, apparent bytes (per-file sum), st_nlink distribution;
  * for every inode with st_nlink > 1, ALL hard-link names via FindFirstFileNameW,
    classified as inside the env, inside the rattler cache, or other;
  * "internal" bytes = inodes whose every link lies inside the env (what deleting the
    env alone frees, and the incremental disk cost of the env when the cache already
    holds the other links). Allocated (on-disk) bytes via GetCompressedFileSizeW;
  * package-set equality between the C: env and the G: project env
    (conda-meta file names, name/version/build/sha256, and a relpath+size manifest);
  * absolute-path leaks to the G: project inside the C: and G: envs;
  * what each env's python imports and whether xgboost is CUDA-enabled
    (run with -I -B so no bytecode or other files are written into any env);
  * a re-walk of the C: env after the import probe to confirm the measurement is stable.

Run with the project interpreter:
  G:\pureskill\csds-cs2\.pixi\envs\default\python.exe -I -B -X utf8 <this file>
Writes only logs\swarm2\verify_E3_3.log.
"""
import ctypes
import json
import os
import re
import stat as statmod
import subprocess
import sys
import time
from collections import Counter
from ctypes import wintypes

LOG = r"G:\pureskill\csds-cs2\logs\swarm2\verify_E3_3.log"
SCRATCH = (r"C:\Users\<user>\AppData\Local\Temp\claude"
           r"\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
CACHE_ROOT = r"C:\Users\<user>\AppData\Local\rattler\cache"
ENVS = [
    ("C_csds_cs2", r"C:\Users\<user>\.pixi\workspace-envs\csds-cs2-15611918389529754563\envs\default"),
    ("G_project", r"G:\pureskill\csds-cs2\.pixi\envs\default"),
    ("C_cs2gpu", r"C:\Users\<user>\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default"),
    ("C_cs2gpu_alt", r"C:\Users\<user>\.pixi\workspace-envs\cs2gpu_alt-16300978804608365045\envs\default"),
    ("C_cs2gpu_torch", r"C:\Users\<user>\.pixi\workspace-envs\cs2gpu_torch-7064157186003075779\envs\default"),
]
CLAIM = {
    "C_csds_cs2": {"files": 20617, "apparent": 1.864, "hardlinked": 20214, "unique": 0.026},
    "C_cs2gpu": {"apparent": 2.097, "unique": 0.043},
    "C_cs2gpu_alt": {"apparent": 1.294, "unique": 0.034},
}
GIB = 1024 ** 3
INVALID_HANDLE = ctypes.c_void_p(-1).value
ERROR_HANDLE_EOF = 38
REPARSE = 0x400
BUF_CHARS = 32768
CUDA_RE = re.compile(r"(cudart|cublas|nvrtc|cudnn|cufft|cusparse|nvjitlink|xgboost_cuda)", re.I)

OUT = []


def say(*parts):
    s = " ".join(str(p) for p in parts)
    OUT.append(s)
    print(s, flush=True)


k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.FindFirstFileNameW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR]
k32.FindFirstFileNameW.restype = ctypes.c_void_p
k32.FindNextFileNameW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR]
k32.FindNextFileNameW.restype = wintypes.BOOL
k32.FindClose.argtypes = [ctypes.c_void_p]
k32.FindClose.restype = wintypes.BOOL
k32.GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
k32.GetCompressedFileSizeW.restype = wintypes.DWORD


def hardlink_names(path):
    """All hard-link names of the file at path (volume-relative or drive-qualified)."""
    size = wintypes.DWORD(BUF_CHARS)
    buf = ctypes.create_unicode_buffer(BUF_CHARS)
    h = k32.FindFirstFileNameW(path, 0, ctypes.byref(size), buf)
    if h is None or h == INVALID_HANDLE:
        raise OSError(ctypes.get_last_error(), "FindFirstFileNameW failed: " + path)
    names = [buf.value]
    try:
        while True:
            size.value = BUF_CHARS
            if not k32.FindNextFileNameW(h, ctypes.byref(size), buf):
                err = ctypes.get_last_error()
                if err == ERROR_HANDLE_EOF:
                    break
                raise OSError(err, "FindNextFileNameW failed: " + path)
            names.append(buf.value)
    finally:
        k32.FindClose(h)
    return names


def alloc_bytes(path):
    high = wintypes.DWORD(0)
    low = k32.GetCompressedFileSizeW(path, ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.get_last_error() != 0:
        return None
    return (high.value << 32) + low


def walk_env(root):
    """Regular files as (path, rel, size, nlink, dev, ino); reparse points listed separately."""
    recs, reparse, errs = [], [], []
    for dp, dns, fns in os.walk(root, followlinks=False):
        keep = []
        for name in dns:
            p = os.path.join(dp, name)
            try:
                st = os.stat(p, follow_symlinks=False)
            except OSError as e:
                errs.append("%s: %s" % (p, e))
                continue
            if getattr(st, "st_file_attributes", 0) & REPARSE or statmod.S_ISLNK(st.st_mode):
                reparse.append(p)
            else:
                keep.append(name)
        dns[:] = keep
        for name in fns:
            p = os.path.join(dp, name)
            try:
                st = os.stat(p, follow_symlinks=False)
            except OSError as e:
                errs.append("%s: %s" % (p, e))
                continue
            if getattr(st, "st_file_attributes", 0) & REPARSE or statmod.S_ISLNK(st.st_mode):
                reparse.append(p)
                continue
            recs.append((p, os.path.relpath(p, root), st.st_size, st.st_nlink, st.st_dev, st.st_ino))
    return recs, reparse, errs


def analyze(label, root):
    t0 = time.time()
    recs, reparse, errs = walk_env(root)
    root_n = os.path.normcase(root).rstrip("\\") + "\\"
    cache_n = os.path.normcase(CACHE_ROOT).rstrip("\\") + "\\"
    drive = root[:2]
    per_inode = Counter((r[4], r[5]) for r in recs)
    inodes = {}
    for r in recs:
        inodes.setdefault((r[4], r[5]), r)
    res = {
        "label": label, "root": root, "files": len(recs), "reparse": len(reparse),
        "walk_errors": len(errs), "errs": errs[:5],
        "nlink1_files": sum(1 for r in recs if r[3] == 1),
        "nlink1_apparent": sum(r[2] for r in recs if r[3] == 1),
        "apparent_all": sum(r[2] for r in recs),
        "inodes": len(inodes), "apparent_dedup": 0, "alloc_dedup": 0,
        "internal": [0, 0, 0], "shared_cache_only": [0, 0, 0], "shared_other": [0, 0, 0],
        "files_with_cache_link": 0, "outside_other": Counter(), "enum_mismatch": 0,
        "env_dupe_inodes": 0, "alloc_errors": 0, "alloc_samples": [],
        "manifest": {}, "cuda_names": set(),
    }
    res["nlinkgt1_files"] = res["files"] - res["nlink1_files"]
    for key, r in inodes.items():
        path, rel, size, nlink = r[0], r[1], r[2], r[3]
        res["manifest"][rel] = size
        if CUDA_RE.search(os.path.basename(path)):
            res["cuda_names"].add(os.path.basename(path))
        res["apparent_dedup"] += size
        if nlink > 1:
            full = []
            for nm in hardlink_names(path):
                full.append(nm if (len(nm) > 1 and nm[1] == ":") else drive + nm)
            if len(full) != nlink:
                res["enum_mismatch"] += 1
            n_env = n_cache = n_other = 0
            for fp in full:
                f = os.path.normcase(fp)
                if f.startswith(root_n):
                    n_env += 1
                elif f.startswith(cache_n):
                    n_cache += 1
                else:
                    n_other += 1
                    res["outside_other"]["\\".join(fp.split("\\")[:5])] += 1
            if n_env > 1:
                res["env_dupe_inodes"] += 1
            if n_cache:
                res["files_with_cache_link"] += per_inode[key]
            if n_cache == 0 and n_other == 0:
                cat = "internal"
            elif n_other == 0:
                cat = "shared_cache_only"
            else:
                cat = "shared_other"
        else:
            cat = "internal"
        a = alloc_bytes(path)
        if a is None:
            res["alloc_errors"] += 1
            a = size
        if size > 0 and len(res["alloc_samples"]) < 3:
            res["alloc_samples"].append((rel, size, a))
        res["alloc_dedup"] += a
        c = res[cat]
        c[0] += 1
        c[1] += size
        c[2] += a
    res["seconds"] = round(time.time() - t0, 1)
    return res


def gib(b):
    return "%.4f GiB (%d B)" % (b / GIB, b)


def report(res):
    say("")
    say("== ENV %s : %s" % (res["label"], res["root"]))
    say("  regular files            : %d   (reparse/symlink entries skipped: %d, walk errors: %d)"
        % (res["files"], res["reparse"], res["walk_errors"]))
    say("  files with nlink == 1    : %d" % res["nlink1_files"])
    say("  files with nlink > 1     : %d" % res["nlinkgt1_files"])
    say("  unique inodes            : %d" % res["inodes"])
    say("  apparent, per-file sum   : %s" % gib(res["apparent_all"]))
    say("  apparent, per-inode sum  : %s" % gib(res["apparent_dedup"]))
    say("  allocated, per-inode sum : %s" % gib(res["alloc_dedup"]))
    for k, title in [("internal", "all links inside env (deleting env frees these)"),
                     ("shared_cache_only", "links in env + rattler cache only"),
                     ("shared_other", "links in env + some other location")]:
        n, ap, al = res[k]
        say("  %-18s inodes=%-7d apparent=%s allocated=%s  [%s]"
            % (k, n, gib(ap), gib(al), title))
    say("  files whose link set includes the rattler cache: %d" % res["files_with_cache_link"])
    say("  hard-link enumeration count mismatches (vs st_nlink): %d" % res["enum_mismatch"])
    say("  inodes with >1 link inside env: %d" % res["env_dupe_inodes"])
    say("  GetCompressedFileSizeW errors: %d" % res["alloc_errors"])
    say("  sample (rel, size, allocated): %s" % res["alloc_samples"])
    if res["outside_other"]:
        say("  top outside-env, non-cache link prefixes:")
        for p, n in res["outside_other"].most_common(8):
            say("    %6d  %s" % (n, p))
    say("  cuda-looking file names: %s" % (sorted(res["cuda_names"])[:12] or "none"))
    say("  elapsed seconds: %s" % res["seconds"])


def conda_meta(root):
    d = os.path.join(root, "conda-meta")
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                j = json.load(fh)
            out[fn] = (j.get("name"), j.get("version"), j.get("build"), j.get("sha256") or j.get("md5"))
        except Exception as e:
            out[fn] = ("ERR", str(e), None, None)
    return out


def ref_scan(root):
    seen, hits, big = set(), [], []
    scanned = skipped = 0
    for dp, dns, fns in os.walk(root, followlinks=False):
        for f in fns:
            p = os.path.join(dp, f)
            try:
                st = os.stat(p, follow_symlinks=False)
            except OSError:
                skipped += 1
                continue
            if getattr(st, "st_file_attributes", 0) & REPARSE:
                skipped += 1
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            if st.st_size > 64 * 1024 * 1024:
                big.append(os.path.relpath(p, root))
                continue
            try:
                with open(p, "rb") as fh:
                    data = fh.read().lower()
            except OSError:
                skipped += 1
                continue
            scanned += 1
            if b"pureskill" in data or "pureskill".encode("utf-16-le") in data:
                hits.append(os.path.relpath(p, root))
    return scanned, skipped, hits, big


def import_probe(py):
    code = (
        "import sys\n"
        "import numpy, polars, lightgbm, xgboost\n"
        "print('sys.executable=' + sys.executable)\n"
        "print('sys.prefix=' + sys.prefix)\n"
        "print('xgboost=' + xgboost.__version__ + ' file=' + xgboost.__file__)\n"
        "try:\n"
        "    bi = xgboost.build_info()\n"
        "    print('xgboost USE_CUDA=' + str(bi.get('USE_CUDA')))\n"
        "except Exception as e:\n"
        "    print('xgboost build_info error: ' + repr(e))\n"
        "print('lightgbm=' + lightgbm.__version__ + ' file=' + lightgbm.__file__)\n"
        "print('polars=' + polars.__version__)\n"
        "print('numpy=' + numpy.__version__)\n"
    )
    try:
        r = subprocess.run([py, "-I", "-B", "-X", "utf8", "-c", code], capture_output=True,
                           text=True, timeout=600, cwd=SCRATCH)
        return r.returncode, r.stdout.strip(), r.stderr.strip()[-1500:]
    except Exception as e:
        return -1, "", repr(e)


def main():
    say("verify_E3_3 recomputation (independent of claimed numbers)")
    say("interpreter: " + sys.executable + " | sys.prefix: " + sys.prefix)
    say("GiB = 1024**3 bytes; claimed values are rounded to 3 decimals, tolerance 0.0005 GiB")

    c_root = ENVS[0][1]
    py_c = os.path.join(c_root, "python.exe")
    st = os.stat(py_c, follow_symlinks=False)
    say("")
    say("== probe: C env python.exe st_nlink=%d st_size=%d" % (st.st_nlink, st.st_size))
    for n in hardlink_names(py_c):
        say("   link: " + n)
    a = alloc_bytes(py_c)
    say("   allocated (GetCompressedFileSizeW) = %s vs apparent %d" % (a, st.st_size))

    results = {}
    for label, root in ENVS:
        if not os.path.isdir(root):
            say("MISSING ENV: " + root)
            continue
        results[label] = analyze(label, root)
        report(results[label])

    say("")
    say("== CLAIM CHECK (C_csds_cs2)")
    r = results["C_csds_cs2"]
    cl = CLAIM["C_csds_cs2"]
    rows = [
        ("files", cl["files"], r["files"], r["files"] == cl["files"]),
        ("apparent GiB", cl["apparent"], round(r["apparent_all"] / GIB, 3),
         abs(r["apparent_all"] / GIB - cl["apparent"]) <= 0.0005),
        ("hard-linked files (nlink>1)", cl["hardlinked"], r["nlinkgt1_files"],
         r["nlinkgt1_files"] == cl["hardlinked"]),
        ("unique GiB (internal, apparent)", cl["unique"], round(r["internal"][1] / GIB, 4),
         abs(r["internal"][1] / GIB - cl["unique"]) <= 0.0005),
    ]
    for name, claimed, got, ok in rows:
        say("  %-34s claimed=%-10s recomputed=%-12s %s" % (name, claimed, got, "MATCH" if ok else "MISMATCH"))
    say("  unique GiB, allocated (internal) = %.4f" % (r["internal"][2] / GIB))
    say("  unique GiB via nlink==1 files only (per-file sum) = %.4f" % (r["nlink1_apparent"] / GIB))
    say("  dedup apparent (per-inode) after cache-shared bytes: %.4f GiB" % (r["apparent_dedup"] / GIB))
    say("  disk freed by deleting env ONLY (internal):          %.4f GiB apparent" % (r["internal"][1] / GIB))
    say("  disk freed by deleting env AND clearing rattler cache: %.4f GiB apparent (= whole env)"
        % (r["apparent_dedup"] / GIB))
    for label in ["C_cs2gpu", "C_cs2gpu_alt"]:
        if label in results:
            rr = results[label]
            cc = CLAIM[label]
            say("  %s: apparent claimed %.3f recomputed %.4f | unique claimed %.3f recomputed %.4f"
                % (label, cc["apparent"], rr["apparent_all"] / GIB, cc["unique"], rr["internal"][1] / GIB))
    if "C_cs2gpu_torch" in results:
        rr = results["C_cs2gpu_torch"]
        say("  C_cs2gpu_torch (not in claim): apparent %.4f GiB, unique %.4f GiB, files %d"
            % (rr["apparent_all"] / GIB, rr["internal"][1] / GIB, rr["files"]))

    say("")
    say("== PACKAGE SET: C_csds_cs2 vs G_project (conda-meta + relpath/size manifest)")
    cm_c = conda_meta(c_root)
    cm_g = conda_meta(ENVS[1][1])
    say("  conda-meta json count: C=%d G=%d" % (len(cm_c), len(cm_g)))
    only_c = sorted(set(cm_c) - set(cm_g))
    only_g = sorted(set(cm_g) - set(cm_c))
    val_diff = [k for k in set(cm_c) & set(cm_g) if cm_c[k] != cm_g[k]]
    say("  conda-meta files only in C: %d %s" % (len(only_c), only_c[:5]))
    say("  conda-meta files only in G: %d %s" % (len(only_g), only_g[:5]))
    say("  name/version/build/sha256 mismatches on shared files: %d %s"
        % (len(val_diff), [(k, cm_c[k], cm_g[k]) for k in sorted(val_diff)[:3]]))
    mc = results["C_csds_cs2"]["manifest"]
    mg = results["G_project"]["manifest"]
    only_mc = sorted(set(mc) - set(mg))
    only_mg = sorted(set(mg) - set(mc))
    size_diff = sorted(k for k in set(mc) & set(mg) if mc[k] != mg[k])
    say("  manifest relpaths: C=%d G=%d" % (len(mc), len(mg)))
    say("  relpaths only in C: %d  sample %s" % (len(only_mc), only_mc[:8]))
    say("  relpaths only in G: %d  sample %s" % (len(only_mg), only_mg[:8]))
    say("  same relpath, different size: %d  sample %s"
        % (len(size_diff), [(k, mc[k], mg[k]) for k in size_diff[:8]]))

    say("")
    say("== ABSOLUTE PATH REFERENCES (pattern 'pureskill', ascii and utf-16le, case-insensitive)")
    for label, root in [("C_csds_cs2", c_root), ("G_project", ENVS[1][1])]:
        scanned, skipped, hits, big = ref_scan(root)
        say("  %s: files scanned=%d skipped=%d over-64MB=%d hits=%d" % (label, scanned, skipped, len(big), len(hits)))
        for h in hits[:25]:
            say("    hit: " + h)

    say("")
    say("== IMPORT PROBE (-I -B; no bytecode written)")
    for label, root in [("C_csds_cs2", c_root), ("G_project", ENVS[1][1])]:
        rc, out, err = import_probe(os.path.join(root, "python.exe"))
        say("  %s rc=%d" % (label, rc))
        for line in out.splitlines():
            say("    " + line)
        if err:
            say("    stderr tail: " + err.replace("\n", " | ")[:800])

    say("")
    say("== RE-WALK C_csds_cs2 after import probe (stability)")
    recs2, _, _ = walk_env(c_root)
    say("  files now=%d (was %d); apparent now=%d (was %d)"
        % (len(recs2), results["C_csds_cs2"]["files"],
           sum(x[2] for x in recs2), results["C_csds_cs2"]["apparent_all"]))

    say("")
    say("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        say("EXCEPTION: " + repr(e))
        say(traceback.format_exc())
    finally:
        with open(LOG, "w", encoding="utf-8") as fh:
            fh.write("\n".join(OUT) + "\n")
