"""verify_E3_1_size.py: independent recount for finding E3-1.

Recomputes, for each directory tree given on the command line:
  - regular file count, apparent bytes (st_size), and on-disk bytes (GetCompressedFileSizeW)
  - hard-link status per file (st_nlink > 1), with example paths
  - device (volume) id per file (st_dev) and the set of distinct devices
  - __pycache__ subtotal (runtime bytecode written after install)
  - symlinks, junctions and other non-regular entries (reported, not followed)

Method differs from E3_size_probe.py on purpose: os.walk with followlinks=False,
junctions pruned and reported, on-disk size added, and symlink/junction entries
reported instead of silently skipped. Read-only.

Usage (project interpreter, explicitly):
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 verify_E3_1_size.py <root> [<root> ...]
"""
import ctypes
import os
import stat
import sys
import time
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
GetCompressedFileSizeW = kernel32.GetCompressedFileSizeW
GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
GetCompressedFileSizeW.restype = wintypes.DWORD
INVALID_FILE_SIZE = 0xFFFFFFFF
NO_ERROR = 0
GIB = 2 ** 30


def on_disk_bytes(path):
    high = wintypes.DWORD(0)
    low = GetCompressedFileSizeW(path, ctypes.byref(high))
    if low == INVALID_FILE_SIZE:
        err = ctypes.get_last_error()
        if err != NO_ERROR:
            raise OSError(err, "GetCompressedFileSizeW failed", path)
    return (high.value << 32) | low


def is_junction(path):
    fn = getattr(os.path, "isjunction", None)
    return bool(fn and fn(path))


def scan(root):
    t0 = time.time()
    res = {
        "root": root,
        "dirs": 0,
        "files": 0,
        "apparent": 0,
        "ondisk": 0,
        "ondisk_errors": 0,
        "stat_errors": 0,
        "nlink_gt1_files": 0,
        "nlink_gt1_bytes": 0,
        "nlink_max": 0,
        "nlink_hist": {},
        "devs": set(),
        "pycache_files": 0,
        "pycache_bytes": 0,
        "symlinks": [],
        "junctions": [],
        "other_nonfile": 0,
        "linked_examples": [],
        "top": {},
    }

    def on_walk_error(exc):
        res["stat_errors"] += 1

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=on_walk_error):
        res["dirs"] += 1
        keep = []
        for name in dirnames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full) or is_junction(full):
                if is_junction(full):
                    res["junctions"].append(full)
                else:
                    res["symlinks"].append(full)
                continue
            keep.append(name)
        dirnames[:] = keep

        rel = os.path.relpath(dirpath, root)
        top = "(root files)" if rel == "." else rel.split(os.sep)[0]
        in_pycache = "__pycache__" in rel.split(os.sep)

        for name in filenames:
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(full, follow_symlinks=False)
            except OSError:
                res["stat_errors"] += 1
                continue
            if stat.S_ISLNK(st.st_mode):
                res["symlinks"].append(full)
                continue
            if not stat.S_ISREG(st.st_mode):
                res["other_nonfile"] += 1
                continue

            res["files"] += 1
            res["apparent"] += st.st_size
            res["devs"].add(st.st_dev)
            nlink = st.st_nlink
            res["nlink_hist"][nlink] = res["nlink_hist"].get(nlink, 0) + 1
            if nlink > res["nlink_max"]:
                res["nlink_max"] = nlink
            if nlink > 1:
                res["nlink_gt1_files"] += 1
                res["nlink_gt1_bytes"] += st.st_size
                if len(res["linked_examples"]) < 5:
                    res["linked_examples"].append(full)
            if in_pycache:
                res["pycache_files"] += 1
                res["pycache_bytes"] += st.st_size

            bucket = res["top"].setdefault(top, [0, 0])
            bucket[0] += 1
            bucket[1] += st.st_size

            try:
                res["ondisk"] += on_disk_bytes(full)
            except OSError:
                res["ondisk_errors"] += 1

    res["secs"] = round(time.time() - t0, 1)
    return res


def report(res):
    print("ROOT %s" % res["root"])
    print("  dirs=%d files=%d apparent_bytes=%d apparent_GiB=%.3f ondisk_GiB=%.3f"
          % (res["dirs"], res["files"], res["apparent"], res["apparent"] / GIB, res["ondisk"] / GIB))
    print("  nlink_gt1_files=%d nlink_gt1_GiB=%.3f nlink_max=%d nlink_hist=%s"
          % (res["nlink_gt1_files"], res["nlink_gt1_bytes"] / GIB, res["nlink_max"],
             dict(sorted(res["nlink_hist"].items()))))
    print("  devices=%s" % ", ".join("0x%08X" % d for d in sorted(res["devs"])))
    print("  pycache_files=%d pycache_GiB=%.4f" % (res["pycache_files"], res["pycache_bytes"] / GIB))
    print("  symlink_entries=%d junction_entries=%d other_nonfile=%d stat_errors=%d ondisk_errors=%d secs=%.1f"
          % (len(res["symlinks"]), len(res["junctions"]), res["other_nonfile"],
             res["stat_errors"], res["ondisk_errors"], res["secs"]))
    for p in res["symlinks"][:10]:
        print("    SYMLINK %s" % p)
    for p in res["junctions"][:10]:
        print("    JUNCTION %s" % p)
    for p in res["linked_examples"]:
        print("    LINKED_EXAMPLE %s" % p)
    for k, v in sorted(res["top"].items(), key=lambda kv: -kv[1][1])[:12]:
        print("  TOP %-30s files=%-7d apparent_GiB=%.3f" % (k, v[0], v[1] / GIB))
    sys.stdout.flush()


def device_of(path):
    try:
        return "0x%08X" % os.stat(path).st_dev
    except OSError as exc:
        return "ERROR %s" % exc


if __name__ == "__main__":
    roots = sys.argv[1:]
    print("interpreter %s" % sys.executable)
    print("python %s" % sys.version.split()[0])
    print("")
    for r in roots:
        report(scan(r))
        print("")
    print("VOLUME IDS (st_dev is the Windows volume serial number)")
    for label, path in [
        ("G: repo root", "G:/pureskill/csds-cs2"),
        ("G: project .pixi", "G:/pureskill/csds-cs2/.pixi"),
        ("C: global .pixi", "C:/Users/<user>/.pixi"),
        ("C: rattler cache", "C:/Users/<user>/AppData/Local/rattler/cache"),
    ]:
        print("  %-20s %-50s %s" % (label, path, device_of(path)))
