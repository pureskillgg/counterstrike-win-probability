"""verify_E3_7 part 1 (read-only): drive facts, free space, same-file-set read timing.

Run with the project interpreter (no bytecode is written; sys.dont_write_bytecode is set below):
  G:/pureskill/csds-cs2/.pixi/envs/default/python.exe -I -X utf8 G:/pureskill/csds-cs2/scripts/swarm2/verify_E3_7_fs.py

Writes nothing to disk. Reads only:
  - the G: project env copy and the C: detached env copy (same package files, Lib/site-packages)
  - the three largest common files in that env
  - one dataframe_*.parquet from a G: tome channel folder (the real data path)
Memory: 1 MiB chunked reads, file contents are discarded immediately.
"""
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True

GIB = 1024.0 ** 3
MIB = 1024.0 ** 2
G_ENV = "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default"
C_ENV = "C:\\Users\\<user>\\.pixi\\workspace-envs\\csds-cs2-15611918389529754563\\envs\\default"
G_TOME = "G:\\pureskill\\csds-cs2\\tomes\\2026-10-01_to_08\\tome\\csds"
SP = os.path.join("Lib", "site-packages")
N_SMALL = 3000
N_BIG = 3
CHUNK = 1 << 20

# Claimed values from the E3-7 finding (GiB unless the claim says GB; PowerShell GB = GiB here).
CLAIM_TOTAL_GIB = {"C:": 1861.9, "G:": 931.5, "E:": 7452.0, "F:": 239.0}
CLAIM_FREE_GIB = {"G:": 207.66, "E:": 5930.54, "F:": 7.95}


def out(s=""):
    print(s, flush=True)


def ps(cmd):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    return (r.stdout or "") + (r.stderr or "")


def section_drives():
    out("== 1. Drive facts (PowerShell Get-Disk / Get-PhysicalDisk / Get-Volume) ==")
    out(ps("Get-Disk | Select-Object Number,FriendlyName,BusType,PartitionStyle,"
           "@{n='SizeGiB';e={[math]::Round($_.Size/1GB,2)}},HealthStatus | Format-Table -AutoSize | Out-String -Width 200"))
    out(ps("Get-PhysicalDisk | Select-Object DeviceId,FriendlyName,MediaType,BusType,HealthStatus | "
           "Format-Table -AutoSize | Out-String -Width 200"))
    out(ps("Get-Volume | Where-Object {$_.DriveLetter} | Select-Object DriveLetter,FileSystem,DriveType,HealthStatus,"
           "@{n='SizeGiB';e={[math]::Round($_.Size/1GB,2)}},@{n='FreeGiB';e={[math]::Round($_.SizeRemaining/1GB,2)}} | "
           "Format-Table -AutoSize | Out-String -Width 200"))


def section_free_space():
    out("== 2. Free space (shutil.disk_usage) vs claim ==")
    for letter in ["C:", "E:", "F:", "G:"]:
        try:
            u = __import__("shutil").disk_usage(letter + "\\")
        except OSError as exc:
            out("  %s ERROR %s" % (letter, exc))
            continue
        total = u.total / GIB
        free = u.free / GIB
        ct = CLAIM_TOTAL_GIB.get(letter)
        cf = CLAIM_FREE_GIB.get(letter)
        out("  %s total=%.2f GiB (claim %s)  free=%.2f GiB (claim %s)"
            % (letter, total, ct if ct is not None else "n/a", free, cf if cf is not None else "n/a"))
    out("  NOTE: task context says C: has about 49 GB free; the measured value is recorded above.")


def walk_sp(root):
    base = os.path.join(root, SP)
    rels = {}
    for dp, dns, fns in os.walk(base):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            full = os.path.join(dp, fn)
            try:
                rels[os.path.relpath(full, base)] = os.stat(full).st_size
            except OSError:
                pass
    return rels


def read_set(paths):
    total = 0
    n = 0
    errors = 0
    t0 = time.perf_counter()
    for p in paths:
        try:
            with open(p, "rb") as fh:
                while True:
                    b = fh.read(CHUNK)
                    if not b:
                        break
                    total += len(b)
            n += 1
        except OSError:
            errors += 1
    return time.perf_counter() - t0, n, total, errors


def fmt(label, res):
    secs, n, total, errors = res
    rate_files = n / secs if secs > 0 else float("nan")
    rate_mb = (total / MIB) / secs if secs > 0 else float("nan")
    out("  %-22s files=%5d bytes=%10.1f MiB  secs=%8.3f  files/s=%9.1f  MiB/s=%8.2f  errors=%d"
        % (label, n, total / MIB, secs, rate_files, rate_mb, errors))


def section_same_file_set():
    out("== 3. Same file set on both copies (Lib/site-packages, __pycache__ excluded) ==")
    t0 = time.perf_counter()
    rg = walk_sp(G_ENV)
    rc = walk_sp(C_ENV)
    out("  walk: G files=%d  C files=%d  walk_secs=%.2f" % (len(rg), len(rc), time.perf_counter() - t0))
    common = sorted(set(rg) & set(rc))
    only_g = len(set(rg) - set(rc))
    only_c = len(set(rc) - set(rg))
    mismatch = [k for k in common if rg[k] != rc[k]]
    out("  common=%d only_G=%d only_C=%d size_mismatch=%d" % (len(common), only_g, only_c, len(mismatch)))

    py = [k for k in common if k.endswith(".py")]
    step = max(1, len(py) // N_SMALL)
    sample = py[::step][:N_SMALL]
    out("  small-file sample: %d .py files (every %d-th of %d)  [ASSUMPTION: .py files are representative small reads]"
        % (len(sample), step, len(py)))
    g_paths = [os.path.join(G_ENV, SP, k) for k in sample]
    c_paths = [os.path.join(C_ENV, SP, k) for k in sample]
    out("  order: G pass1, C pass1, C pass2, G pass2  (cache state NOT controlled; no admin to drop caches)")
    out("  -- small files --")
    fmt("G pass1", read_set(g_paths))
    fmt("C pass1", read_set(c_paths))
    fmt("C pass2", read_set(c_paths))
    fmt("G pass2", read_set(g_paths))

    big = sorted(common, key=lambda k: rg[k], reverse=True)[:N_BIG]
    out("  -- largest common files (sequential) --")
    for k in big:
        out("    %s  %.1f MiB" % (k, rg[k] / MIB))
    gb = [os.path.join(G_ENV, SP, k) for k in big]
    cb = [os.path.join(C_ENV, SP, k) for k in big]
    fmt("G pass1 big", read_set(gb))
    fmt("C pass1 big", read_set(cb))
    fmt("C pass2 big", read_set(cb))
    fmt("G pass2 big", read_set(gb))


def section_parquet():
    out("== 4. Real data path: one tome parquet on G: (no C: copy exists for comparison) ==")
    try:
        chans = sorted(d for d in os.listdir(G_TOME) if d.startswith("basic_"))
    except OSError as exc:
        out("  cannot list tome: %s" % exc)
        return
    if not chans:
        out("  no basic_ channel folders found")
        return
    ch_dir = os.path.join(G_TOME, chans[0])
    pq = [f for f in os.listdir(ch_dir) if f.startswith("dataframe_") and f.endswith(".parquet")]
    if not pq:
        out("  no dataframe_*.parquet in %s" % chans[0])
        return
    best = max(pq, key=lambda f: os.stat(os.path.join(ch_dir, f)).st_size)
    p = os.path.join(ch_dir, best)
    out("  file: %s  %.1f MiB" % (p, os.stat(p).st_size / MIB))
    fmt("G parquet pass1", read_set([p]))
    fmt("G parquet pass2", read_set([p]))


def main():
    out("verify_E3_7_fs.py  interpreter=%s" % sys.executable)
    out("sys.prefix=%s" % sys.prefix)
    section_drives()
    section_free_space()
    section_same_file_set()
    section_parquet()
    out("== done ==")


if __name__ == "__main__":
    main()
