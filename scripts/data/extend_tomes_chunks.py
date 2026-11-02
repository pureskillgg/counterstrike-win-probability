"""Extend tome collections day by day, in chunks of --chunk matches, one build at a time.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\data\\extend_tomes_chunks.py --days 2026-10-14,2026-10-15 [--chunk 500] [--dry-run]

This is the chunked queue of run_tome_queue.ps1 without PowerShell, and without its slow per-run scan: the day folders are listed once,
and each chunk is one build_tomes_incremental.py run with --skip-keys (every other match of the day) and --no-check-complete (the 43-file
rule below does the same job as the per-channel check, which is what made the old scan take 443 s).

Per day name D (YYYY-MM-DD): tome_ledger.py says which matches are already inside a finished tome of that name. The day's complete match
folders on E: that are not built yet are cut into chunks; each chunk is built as tomes/D_cNN (BUILD_OK when done), so a crash costs one
chunk. When the day has no unbuilt match, tomes/D.CHUNKS_DONE lists its finished collections (not written while any folder is half-synced).
Nothing is written under E:. tomes/D.LOCK holds the owner's pid so two runs cannot build the same day. logs\\STOP_QUEUE stops after the
current chunk.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("E:/csds_final")
SRC = RAW / "csds"
TOMES = ROOT / "tomes"
LOGS = ROOT / "logs"
PY_TOOLS = ROOT / ".pixi" / "envs" / "default" / "python.exe"
PY_BUILD = Path("G:/pureskill/envs/dsdk-fork/Scripts/python.exe")
BUILDER = ROOT / "scripts" / "data" / "build_tomes_incremental.py"
LEDGER = ROOT / "scripts" / "data" / "tome_ledger.py"
MIN_FILES = 43  # every complete match folder has 43 objects (run_tome_queue.ps1)
RETRIES = 3


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def stop_requested():
    return (LOGS / "STOP_QUEUE").exists()


def ledger(name):
    """Matches of this name already inside a finished tome ('dd/id' lines), and the highest chunk number in use."""
    out = subprocess.run(
        [str(PY_TOOLS), "-I", "-X", "utf8", str(LEDGER), "--name", name, "--tomes", str(TOMES)],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout
    built, top = set(), 0
    for ln in out.splitlines():
        if ln.startswith("#chunks"):
            top = int(ln.split()[1])
        elif ln.strip():
            built.add(ln.strip())
    return built, top


def day_folders(yyyy, mm, dd):
    """(complete ids, incomplete ids) of the day's match folders on E:. One listing per folder, as in the queue."""
    base = SRC / yyyy / mm / dd
    complete, incomplete = [], []
    for e in os.scandir(base):
        if not e.is_dir():
            continue
        n_files = sum(1 for f in os.scandir(e.path) if f.is_file())
        (complete if n_files >= MIN_FILES else incomplete).append(e.name)
    return sorted(complete), sorted(incomplete)


def write_chunks_done(name):
    names = []
    if (TOMES / name / "BUILD_OK").is_file():
        names.append(name)
    chunks = sorted(p.name for p in TOMES.glob(f"{name}_c*") if p.is_dir() and (p / "BUILD_OK").is_file())
    names += chunks
    if not names:
        return
    tmp = TOMES / f"{name}.CHUNKS_DONE.tmp"
    tmp.write_text("\n".join(names) + "\n", encoding="utf-8")
    os.replace(tmp, TOMES / f"{name}.CHUNKS_DONE")
    say(f"{name}: CHUNKS_DONE lists {len(names)} collections")


def take_lock(name):
    lock = TOMES / f"{name}.LOCK"
    if lock.exists():
        try:
            owner = int(lock.read_text().strip() or 0)
        except ValueError:
            owner = 0
        if owner and _alive(owner):
            return False
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _alive(pid):
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in r.stdout


def release_lock(name):
    (TOMES / f"{name}.LOCK").unlink(missing_ok=True)


def build_chunk(out_name, day_keys, chunk_keys, day):
    """One chunk: skip every match of the day that is not in this chunk, then build. Retries a failed chunk."""
    out = TOMES / out_name
    skip = TOMES / f"{out_name}.skip.txt"
    skip.write_text("\n".join(k for k in day_keys if k not in chunk_keys) + "\n", encoding="utf-8")
    log = LOGS / f"extend_{out_name}.log"
    for attempt in range(1, RETRIES + 1):
        with open(log, "w", encoding="utf-8") as lf:
            rc = subprocess.run(
                [str(PY_BUILD), "-I", "-X", "utf8", "-u", str(BUILDER), "--raw", str(RAW), "--out", str(out),
                 "--days", day.replace("-", "/"), "--skip-keys", str(skip), "--no-check-complete"],
                stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT,
            ).returncode
        if rc == 0 and (out / "BUILD_OK").is_file():
            skip.unlink(missing_ok=True)
            return True
        say(f"{out_name}: attempt {attempt} failed (exit {rc}); log {log}")
        time.sleep(30)
    return False


def extend_day(day, chunk, dry_run):
    yyyy, mm, dd = day.split("-")
    name = day
    if not dry_run and not take_lock(name):
        say(f"{name}: another run holds {name}.LOCK; skipping")
        return False
    try:
        if not (SRC / yyyy / mm / dd).is_dir():
            say(f"{name}: {SRC / yyyy / mm / dd} missing on E:; skipping")
            return False
        if not dry_run:
            # A chunk folder without BUILD_OK is left by a crash, a power cut or a STOP_QUEUE pause. Its matches are not in
            # the ledger, so they are rebuilt; the half-written folder is moved aside (never deleted) before the ledger is read.
            stale = TOMES / "_partial" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
            for p in sorted(TOMES.glob(f"{name}_c*")):
                if p.is_dir() and not (p / "BUILD_OK").is_file():
                    stale.mkdir(parents=True, exist_ok=True)
                    p.rename(stale / p.name)
                    say(f"{p.name}: no BUILD_OK (crash or pause); moved to {stale / p.name}")
        built, top = ledger(name)
        complete, incomplete = day_folders(yyyy, mm, dd)
        day_keys = [f"csds/{yyyy}/{mm}/{dd}/{i}/csds" for i in complete]
        todo = [i for i in complete if f"{dd}/{i}" not in built]
        chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
        say(f"{name}: {len(complete)} complete, {len(incomplete)} half-synced, {len(built)} already built, "
            f"{len(todo)} to build in {len(chunks)} chunks (next chunk number {top + 1})")
        if dry_run:
            for k, c in enumerate(chunks, start=1):
                say(f"  would build {name}_c{top + k:02d}: {len(c)} matches")
            return True
        for k, ids in enumerate(chunks, start=1):
            if stop_requested():
                say(f"{name}: logs\\STOP_QUEUE exists; stopping before the next chunk")
                return False
            out_name = f"{name}_c{top + k:02d}"
            if (TOMES / out_name / "BUILD_OK").is_file():
                continue
            chunk_keys = {f"csds/{yyyy}/{mm}/{dd}/{i}/csds" for i in ids}
            t0 = time.time()
            ok = build_chunk(out_name, day_keys, chunk_keys, day)
            say(f"{out_name}: {'built' if ok else 'FAILED'} {len(ids)} matches in {time.time() - t0:.0f} s")
            if not ok:
                return False
        if incomplete:
            say(f"{name}: {len(incomplete)} half-synced folders; CHUNKS_DONE not written for this day")
        else:
            write_chunks_done(name)
        return True
    finally:
        if not dry_run:
            release_lock(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True, help="comma-separated YYYY-MM-DD, built in the order given")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    TOMES.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    for day in [d.strip() for d in a.days.split(",") if d.strip()]:
        if stop_requested():
            say("logs\\STOP_QUEUE exists; not starting")
            break
        if not extend_day(day, a.chunk, a.dry_run):
            say(f"{day}: stopped; fix the cause and run again (finished chunks are kept)")
            break


if __name__ == "__main__":
    main()
