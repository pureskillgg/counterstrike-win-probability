"""verify_E3_7_inventory.py -- independent recomputation of the E3-7 storage facts.

Run with the project interpreter (stdlib only):
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 -B verify_E3_7_inventory.py

Read-only with respect to G:, C:, E:, F:. The only writes are a scratch dir under the
system temp dir (C:), removed at the end. Does not read .env or meridian.toml.
"""
import ctypes
import os
import shutil
import subprocess
import tempfile
import time
from ctypes import wintypes

GiB = 1024 ** 3
GB = 1000 ** 3

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetDiskFreeSpaceExW.argtypes = [
    wintypes.LPCWSTR,
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong),
]
k32.GetDiskFreeSpaceExW.restype = wintypes.BOOL
k32.GetVolumeInformationW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
    wintypes.LPWSTR,
    wintypes.DWORD,
]
k32.GetVolumeInformationW.restype = wintypes.BOOL

# winnt.h volume feature flags (FILE_SUPPORTS_HARD_LINKS = 0x00400000)
FS_FLAGS = {
    0x00000001: "CASE_SENSITIVE",
    0x00000010: "COMPRESSION",
    0x00000040: "SPARSE_FILES",
    0x00000080: "REPARSE_POINTS",
    0x00400000: "HARD_LINKS",
}
FILE_ATTRIBUTE_REPARSE_POINT = 0x400

ENV_ON_G = r"G:\pureskill\csds-cs2\.pixi\envs\default"
ENV_ON_C = r"C:\Users\13144\.pixi\workspace-envs\csds-cs2-15611918389529754563\envs\default"
TOME_2026 = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
TOME_2024 = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"


def volume_facts(letter):
    root = letter + ":\\"
    free_caller = ctypes.c_ulonglong(0)
    total = ctypes.c_ulonglong(0)
    free_total = ctypes.c_ulonglong(0)
    ok = k32.GetDiskFreeSpaceExW(root, ctypes.byref(free_caller), ctypes.byref(total), ctypes.byref(free_total))
    if not ok:
        return None
    label = ctypes.create_unicode_buffer(256)
    fsname = ctypes.create_unicode_buffer(256)
    serial = wintypes.DWORD(0)
    maxlen = wintypes.DWORD(0)
    flags = wintypes.DWORD(0)
    k32.GetVolumeInformationW(root, label, 256, ctypes.byref(serial), ctypes.byref(maxlen),
                              ctypes.byref(flags), fsname, 256)
    flag_names = [name for bit, name in FS_FLAGS.items() if flags.value & bit]
    return {
        "letter": letter,
        "label": label.value,
        "fs": fsname.value,
        "total_bytes": total.value,
        "free_bytes": free_total.value,
        "total_GiB": total.value / GiB,
        "total_GB": total.value / GB,
        "free_GiB": free_total.value / GiB,
        "free_GB": free_total.value / GB,
        "flags": flag_names,
    }


def print_volumes():
    print("== 1. Volume facts recomputed with GetDiskFreeSpaceExW / GetVolumeInformationW")
    facts = {}
    for letter in "CDEFG":
        f = volume_facts(letter)
        if f is None:
            print(f"  {letter}: not ready / no media")
            continue
        facts[letter] = f
        print(
            f"  {letter}: label={f['label']!r} fs={f['fs']} total={f['total_GiB']:.3f} GiB "
            f"({f['total_GB']:.1f} GB) free={f['free_GiB']:.3f} GiB ({f['free_GB']:.2f} GB) "
            f"flags={f['flags']}"
        )
    return facts


def check_claims(facts):
    print("== 2. Claimed numbers vs recomputed (claims from the E3-7 verdict and task context)")
    claims = [
        ("G", "total", "931.5 GiB", facts["G"]["total_GiB"], 931.5),
        ("G", "free", "207.66 GiB", facts["G"]["free_GiB"], 207.66),
        ("C", "total", "1861.9 GiB", facts["C"]["total_GiB"], 1861.9),
        ("C", "free", "about 49 GB (task context)", facts["C"]["free_GB"], 49.0),
        ("E", "total", "7452 'GB' (task text)", facts["E"]["total_GiB"], 7452.0),
        ("E", "free", "5930.54 GiB", facts["E"]["free_GiB"], 5930.54),
        ("F", "total", "239 GB", facts["F"]["total_GB"], 239.0),
        ("F", "free", "7.95 GiB", facts["F"]["free_GiB"], 7.95),
    ]
    for letter, what, claimed_txt, measured, claimed_num in claims:
        print(f"  {letter} {what}: claimed {claimed_txt}; measured {measured:.3f} "
              f"(diff vs claimed number {measured - claimed_num:+.3f})")


def link_capability_test():
    print("== 3. Link capability test in a scratch dir on the system temp volume (C:)")
    tmpdir = tempfile.mkdtemp(prefix="e3_7_links_")
    try:
        target = os.path.join(tmpdir, "target.bin")
        with open(target, "wb") as fh:
            fh.write(b"x" * 1024)
        hl = os.path.join(tmpdir, "hardlink.bin")
        try:
            os.link(target, hl)
            print(f"  hardlink on C: OK, st_nlink(target)={os.stat(target).st_nlink}")
        except OSError as e:
            print(f"  hardlink on C: FAILED winerror={getattr(e, 'winerror', None)} {e}")
        sl = os.path.join(tmpdir, "symlink.bin")
        try:
            os.symlink(target, sl)
            print("  symlink on C: OK (process holds SeCreateSymbolicLinkPrivilege or Developer Mode)")
        except OSError as e:
            print(f"  symlink on C: FAILED winerror={getattr(e, 'winerror', None)} {e}")
        jn = os.path.join(tmpdir, "junction")
        rc = subprocess.run(["cmd", "/c", "mklink", "/J", jn, tmpdir],
                            capture_output=True, text=True).returncode
        print(f"  junction (mklink /J) on C: rc={rc} exists={os.path.exists(jn)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    dev_mode = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock") as k:
            dev_mode = winreg.QueryValueEx(k, "AllowDevelopmentWithoutDevLicense")[0]
    except OSError:
        dev_mode = "value not present"
    print(f"  AllowDevelopmentWithoutDevLicense (Developer Mode flag): {dev_mode}")


def walk_env(root):
    """Return (summary dict, {relpath: (size, nlink, reparse)}) for a directory tree."""
    entries = {}
    t0 = time.perf_counter()
    nfiles = ndirs = nlink_gt1 = nreparse = errs = total_bytes = 0
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                children = list(it)
        except OSError:
            errs += 1
            continue
        for e in children:
            try:
                if e.is_dir(follow_symlinks=False):
                    ndirs += 1
                    stack.append(e.path)
                    continue
                st = os.stat(e.path, follow_symlinks=False)
                nfiles += 1
                total_bytes += st.st_size
                reparse = bool(st.st_file_attributes & FILE_ATTRIBUTE_REPARSE_POINT)
                if st.st_nlink > 1:
                    nlink_gt1 += 1
                if reparse:
                    nreparse += 1
                entries[os.path.relpath(e.path, root)] = (st.st_size, st.st_nlink, reparse)
            except OSError:
                errs += 1
    summary = {
        "root": root, "files": nfiles, "dirs": ndirs, "bytes": total_bytes,
        "GiB": total_bytes / GiB, "nlink_gt1": nlink_gt1, "reparse": nreparse,
        "errors": errs, "secs": time.perf_counter() - t0,
    }
    return summary, entries


def print_env_compare():
    print("== 4. Env copies: G: .pixi\\envs\\default vs C: detached workspace-envs copy")
    sg, eg = walk_env(ENV_ON_G)
    sc, ec = walk_env(ENV_ON_C)
    for s in (sg, sc):
        print(f"  {s['root']}: files={s['files']} dirs={s['dirs']} GiB={s['GiB']:.3f} "
              f"nlink>1={s['nlink_gt1']} reparse={s['reparse']} errors={s['errors']} secs={s['secs']:.1f}")
    common = set(eg) & set(ec)
    size_mismatch = [k for k in common if eg[k][0] != ec[k][0]]
    only_g = len(set(eg) - set(ec))
    only_c = len(set(ec) - set(eg))
    print(f"  COMPARE relpaths common={len(common)} size_mismatch={len(size_mismatch)} "
          f"only_on_G={only_g} only_on_C={only_c}")
    print(f"  G copy files with st_nlink>1 (hard-linked): {sg['nlink_gt1']} of {sg['files']}; "
          f"C copy: {sc['nlink_gt1']} of {sc['files']}")
    return eg


def tome_bytes(root):
    total = 0
    n = 0
    t0 = time.perf_counter()
    for channel in os.scandir(root):
        if not channel.is_dir(follow_symlinks=False):
            continue
        for match in os.scandir(channel.path):
            if not match.is_dir(follow_symlinks=False):
                continue
            for f in os.scandir(match.path):
                if f.name.startswith("dataframe_") and f.is_file(follow_symlinks=False):
                    total += f.stat(follow_symlinks=False).st_size
                    n += 1
    return n, total, time.perf_counter() - t0


def print_tomes():
    print("== 5. Tome dataframe bytes on G: (for scale of data reads)")
    for label, root in (("2026-10-01_to_08", TOME_2026), ("2024-02-g", TOME_2024)):
        try:
            n, total, secs = tome_bytes(root)
            print(f"  {label}: dataframe files={n} GiB={total / GiB:.3f} walk_secs={secs:.1f}")
        except OSError as e:
            print(f"  {label}: walk failed {e}")


def main():
    facts = print_volumes()
    check_claims(facts)
    link_capability_test()
    print_env_compare()
    print_tomes()
    print("== done")


if __name__ == "__main__":
    main()
