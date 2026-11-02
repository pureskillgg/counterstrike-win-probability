"""Build or update a tome collection from a folder of CSDS matches, never starting over: fresh, resume or extend, whichever applies.

  python -I -X utf8 scripts\\data\\build_tomes_incremental.py --raw <folder that holds csds/YYYY/MM/DD/<id>/> --out <tome collection folder>
         [--channels a,b,c] [--include-per-tick] [--dry-run]

Run it with the Python of the dsdk fork (G:\\pureskill\\envs\\dsdk-fork, the `incremental-tomes` branch of G:\\pureskill\\dsdk-fork); upstream dsdk 4.1.1
has neither `resume` nor `extend`. Upstream behaviour is described in notes\\TOME_INCREMENTAL_BUILD_DESIGN.md, the fork in its docs/incremental-tomes.md.

What it does, by what it finds in --out:

  nothing there                       fresh   build the header and one tome per channel from all the matches on disk
  an interrupted build (.building)    resume  continue it: keep the finished pages, read only the matches and channels they lack
  a finished collection               extend  read ONLY the matches on disk that the header lacks and append them to the header and to every
                                              channel tome (new pages after the last one; an old page is rewritten only when the new matches
                                              widen a column type). Nothing new on disk = nothing is read or written.

It is idempotent: run it again after a crash, a reboot, or after more matches were synced. Tomes are named basic_<channel> (no dates in the
name, which would change with every extension) and the header is `header`; build_combined.py / append_combined.py read such a collection
(they find tomes by the basic_ prefix and the pages by glob). A BUILD_OK marker is written when a run finishes and removed when one starts
that changes the collection, so a collection that a stopped run left in between is not taken for a finished one.

Safety: it never writes under --raw (--out may not be inside it); a lock file in --out (.incremental.lock) stops a second run on the same
collection; matches whose header or wanted channel files are not all on disk yet (a half-synced folder) are left for a later run
(--no-check-complete to skip that check).
--dry-run prints the plan and writes nothing. Memory: pages are cut at --max-page-size-mb (default 32) per channel and matches are read by
--read-threads (default 2), which keeps the process well under 3 GB for the non-per-tick channels.
"""
import argparse
import gzip
import json
import logging
import os
import sys
import time
import warnings

import structlog

DS_TYPE = "csds"
HEADER = "header"
TOME_NAME = "basic_{channel}"
PER_TICK = {"player_vector", "player_status", "tick", "grenade_vector", "world_item_vector"}
MARKER = "BUILD_OK"
LOCK = ".incremental.lock"
RECORD = "INCREMENTAL.json"
# library events worth a line; warnings and errors always are
SHOW = {
    "Build Basic Tomes Update", "Resuming staged tome", "Extend Basic Tomes", "Extend Basic Tomes Done", "Build Basic Tomes Done",
    "Duplicate match keys dropped", "Orphan page files removed", "Page narrowed", "Channel in no matches: no tome",
}

sys.stdout.reconfigure(encoding="utf-8")


def lib_log(_logger, method, event_dict):
    event = event_dict.get("event")
    if method in ("warning", "error", "critical") or event in SHOW:
        extras = " ".join(f"{k}={v}" for k, v in event_dict.items() if k not in ("event", "level", "timestamp"))
        return f"  [dsdk {method}] {event} {extras}".rstrip()
    raise structlog.DropEvent


def say(phase, text):
    print(f"[{phase}] {text}", flush=True)


def inside(path, folder):
    path, folder = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(folder))
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:  # different drives (the raw data on E:, the tomes on G:): one cannot be inside the other
        return False


def pid_alive(pid):
    if os.name == "nt":  # os.kill(pid, 0) would TERMINATE the process on Windows
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(out):
    path = os.path.join(out, LOCK)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = int(open(path, encoding="utf-8").read().strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and pid_alive(owner):
                sys.exit(f"error: another run (process {owner}) holds {path}")
            os.remove(path)  # stale: its process is gone
            continue
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        return path
    sys.exit(f"error: could not take {path}")


def read_json(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def first_manifest(root):
    """The first manifest under root, found without listing the whole collection."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if DS_TYPE in filenames and os.path.isfile(os.path.join(dirpath, DS_TYPE)):
            return os.path.join(dirpath, DS_TYPE)
    return None


def match_complete(raw, key, needed):
    """A match is complete when the header and every wanted channel file its manifest lists are on disk (a half-synced folder would abort
    the build). Channels that are not built (the per-tick ones, by default) are not looked at."""
    try:
        manifest = read_json(os.path.join(raw, *key.split("/")))
        return all(os.path.isfile(os.path.join(raw, os.path.normpath(e["key"]))) for e in manifest.get("channels", []) if e["channel"] in needed)
    except (OSError, ValueError, KeyError):
        return False


def write_atomically(path, text):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_tome_pages(base, name):
    """[(number, dataframe file, keyset file)] of a channel tome, from its manifest (the ledger), whatever else is in the folder."""
    path = os.path.join(base, name, "tome")
    if not os.path.isfile(path):
        return []
    folder = os.path.dirname(path)
    return [(p["number"], os.path.join(folder, os.path.basename(p["dataframe"]["key"])), os.path.join(folder, os.path.basename(p["keyset"]["key"])))
            for p in read_json(path).get("pages", [])]


def link_or_copy(src, dst):
    import shutil
    try:
        os.link(src, dst)  # pages are immutable once listed: a hard link costs nothing and cannot drift
    except OSError:
        shutil.copy2(src, dst)


def publish_chunk(out, base, names, channels):
    """Publish what the collection gained since the last publish as a small collection of its own, <out>_xNNN, under the same tomes folder,
    and list it in <out>.CHUNKS_DONE. build_combined.py / append_combined.py read a day built in chunks exactly so (a chunk = a source with a
    BUILD_OK marker), so the combined store takes ONLY the new matches, however many times the collection is extended. The chunk holds hard links
    to the new page files of each channel (page numbers above the last published one) and a header page with just the new matches' rows.
    State: <out>/PUBLISHED.json (page watermark per channel) and PUBLISHED_KEYS.txt (the matches already published), written last, so a stop
    in between only means the same chunk is published again on the next run."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    tomes_root, live = os.path.dirname(out), os.path.basename(out)
    state_path, keys_path = os.path.join(out, "PUBLISHED.json"), os.path.join(out, "PUBLISHED_KEYS.txt")
    state = json.load(open(state_path, encoding="utf-8")) if os.path.isfile(state_path) else {"chunks": [], "watermark": {}}
    published = {ln.strip() for ln in open(keys_path, encoding="utf-8") if ln.strip()} if os.path.isfile(keys_path) else set()
    header_pages = read_tome_pages(base, HEADER)
    hdr = pa.concat_tables([pq.read_table(p[1]) for p in header_pages])
    key_col = "key" if "key" in hdr.column_names else hdr.column_names[0]
    all_keys = hdr[key_col].to_pylist()
    new_keys = [k for k in all_keys if k not in published]
    if not new_keys:
        say("publish", "nothing new to publish")
        return None
    n = len(state["chunks"]) + 1
    name = f"{live}_x{n:03d}"
    chunk = os.path.join(tomes_root, name)
    if os.path.exists(chunk):
        import shutil
        shutil.rmtree(chunk)  # a chunk a stopped run left half made (no BUILD_OK): made again from the same state
    csds = os.path.join(chunk, "tome", DS_TYPE)
    hdir = os.path.join(csds, HEADER)
    os.makedirs(hdir)
    pq.write_table(hdr.filter(pc.is_in(hdr[key_col], value_set=pa.array(new_keys))), os.path.join(hdir, "dataframe_00000"))
    pl_keys = pa.table({"_": pa.array(new_keys)})
    pq.write_table(pl_keys, os.path.join(hdir, "keyset_00000"))
    marks, n_pages = dict(state["watermark"]), 0
    for channel in channels:
        pages = [p for p in read_tome_pages(base, names[channel]) if p[0] > state["watermark"].get(channel, -1)]
        if not pages:
            continue
        cdir = os.path.join(csds, names[channel])
        os.makedirs(cdir, exist_ok=True)
        for number, df, ks in pages:
            link_or_copy(df, os.path.join(cdir, os.path.basename(df)))
            link_or_copy(ks, os.path.join(cdir, os.path.basename(ks)))
        marks[channel] = max(p[0] for p in pages)
        n_pages += len(pages)
    write_atomically(os.path.join(chunk, MARKER), f"chunk of {live}: {len(new_keys)} matches, {n_pages} channel pages, {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    done = os.path.join(tomes_root, live + ".CHUNKS_DONE")
    listed = [ln.strip() for ln in open(done, encoding="utf-8") if ln.strip()] if os.path.isfile(done) else []
    if name not in listed:
        write_atomically(done, "\n".join(listed + [name]) + "\n")
    state["chunks"].append({"name": name, "matches": len(new_keys), "pages": n_pages, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    state["watermark"] = marks
    write_atomically(keys_path, "\n".join(sorted(published | set(new_keys))) + "\n")
    write_atomically(state_path, json.dumps(state, indent=1))
    say("publish", f"{name}: {len(new_keys)} new matches, {n_pages} new channel pages (hard links) -> listed in {live}.CHUNKS_DONE; "
        f"merge into the combined store with: append_combined.py --sources {live}")
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", help="only list these upload days of --raw, YYYY/MM/DD comma separated (default: every match folder under --raw)")
    ap.add_argument("--skip-keys", help="text file with one match key per line: matches that are already in another store (the combined one) and must not be built again")
    ap.add_argument("--publish-chunks", action="store_true", help="after the run, publish the new pages as <out>_xNNN and list it in <out>.CHUNKS_DONE for append_combined.py")
    ap.add_argument("--raw", required=True, help="the CSDS collection: the folder that holds csds/YYYY/MM/DD/<id>/ (only read)")
    ap.add_argument("--out", required=True, help="the tome collection folder to create or update")
    ap.add_argument("--channels", help="comma-separated channels (default: every channel of the first match but the per-tick ones)")
    ap.add_argument("--include-per-tick", action="store_true", help="also build the per-tick channels (very large)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--max-page-size-mb", type=float, default=32, help="page size limit per channel in MB of Arrow buffers (default 32)")
    ap.add_argument("--read-threads", type=int, default=2, help="threads that read matches ahead (default 2)")
    ap.add_argument("--no-check-complete", action="store_true", help="do not check that every channel file of a new match is on disk")
    args = ap.parse_args()

    raw, out = os.path.abspath(args.raw), os.path.abspath(args.out)
    if not os.path.isdir(raw):
        sys.exit(f"error: --raw {raw} is not a folder")
    if inside(out, raw) or inside(raw, out):
        sys.exit("error: --out and --raw may not contain one another (nothing is ever written under --raw)")

    warnings.filterwarnings("ignore", message="Header name of")  # tome names without dates are the point
    structlog.configure(processors=[structlog.processors.add_log_level, lib_log], logger_factory=structlog.PrintLoggerFactory())
    from pureskillgg_dsdk.tome import TomeCuratorFs
    from pureskillgg_dsdk.tome.header_tome import get_manifest_key_paths_from_glob

    t_start = time.time()
    if args.channels:
        channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    else:
        first = first_manifest(raw)
        if first is None:
            sys.exit(f"error: no '{DS_TYPE}' manifest found under {raw}")
        channels = [c["channel"] for c in read_json(first)["channels"] if c["channel"] != "header"]
        if not args.include_per_tick:
            channels = [c for c in channels if c not in PER_TICK]
    names = {c: TOME_NAME.format(channel=c) for c in channels}

    # scan: what is on disk, and what the collection holds
    base = os.path.join(out, "tome", DS_TYPE)
    header_manifest_path = os.path.join(base, HEADER, "tome")
    header = read_json(header_manifest_path) if os.path.isfile(header_manifest_path) else None
    complete = bool(header and header.get("isComplete"))
    staged = os.path.isdir(os.path.join(base, ".building", HEADER)) and bool(os.listdir(os.path.join(base, ".building", HEADER)))
    curator = TomeCuratorFs(default_header_name=HEADER, ds_type=DS_TYPE, ds_collection_root_path=raw, tome_collection_root_path=out)
    if args.days:
        # only these upload days are listed (YYYY/MM/DD, comma separated): listing every match folder of a USB drive that holds months of data and is being
        # written by a sync took 38+ minutes; a day folder lists in seconds
        import glob as _glob
        on_disk = []
        for day in [d.strip().strip("/") for d in args.days.split(",") if d.strip()]:
            hits = _glob.glob(os.path.join(raw, DS_TYPE, *day.split("/"), "*", DS_TYPE))
            on_disk += [os.path.relpath(h, raw).replace(os.sep, "/") for h in hits]
        on_disk = sorted(set(on_disk))
    else:
        on_disk = [p.replace(os.sep, "/") for p in get_manifest_key_paths_from_glob(raw, DS_TYPE)]
    in_header = set(curator.get_keyset(HEADER)) if complete else set()
    skip = set()
    if args.skip_keys:
        with open(args.skip_keys, encoding="utf-8") as f:
            skip = {ln.strip() for ln in f if ln.strip()}
    present = in_header | skip
    new = [k for k in on_disk if k not in present]
    already = len(on_disk) - len(new)
    gone = len(in_header - set(on_disk))
    if args.no_check_complete or not new:
        buildable, incomplete = new, []
    else:
        needed = set(channels) | {"header"}
        buildable = [k for k in new if match_complete(raw, k, needed)]
        ok = set(buildable)
        incomplete = [k for k in new if k not in ok]
    say("scan", f"{len(on_disk)} matches on disk in {raw}: {already} already in the header, {len(new)} new"
        f" ({len(buildable)} complete{f', {len(incomplete)} half-synced and left for a later run' if incomplete else ''})"
        f"{f'; {gone} header matches are no longer on disk (kept)' if gone else ''} [{time.time() - t_start:.0f} s]")

    # plan
    record_path = os.path.join(out, RECORD)
    known_channels = set()
    if os.path.isfile(record_path):
        try:
            known_channels = set(read_json(record_path).get("channels", []))
        except (OSError, ValueError):
            pass
    def tome_ok(channel):
        path = os.path.join(base, names[channel], "tome")
        if not os.path.isfile(path):
            return channel in known_channels  # a channel no match has gets no tome
        try:
            return bool(read_json(path).get("isComplete"))
        except (OSError, ValueError):
            return False

    missing = [c for c in channels if not tome_ok(c)]  # no tome yet, or a partial one (a stopped publish): built over all matches
    if complete:
        if not buildable and not missing:
            mode = "nothing"
        else:
            mode = "extend"
    else:
        mode = "resume" if staged else "fresh"
    detail = {
        "nothing": "the header holds every complete match on disk: nothing to read or write",
        "extend": f"append {len(buildable)} new matches to the {len(present)} in the header"
                  + (f"; build the tome of {missing} over all matches" if missing else ""),
        "resume": f"continue the interrupted build in {os.path.join(base, '.building')} over {len(buildable)} matches",
        "fresh": f"build the header and {len(channels)} channel tomes from {len(buildable)} matches",
    }[mode]
    say("plan", f"mode={mode}: {detail}; channels={len(channels)}")
    if mode in ("fresh", "resume") and not buildable:
        # nothing to read yet (a half-synced folder is left for a later run): not an error, run it again when more files have arrived
        say("done", f"nothing to build yet ({len(new)} new match folders on disk, {len(incomplete)} of them not completely synced); run again later [{time.time() - t_start:.0f} s]")
        return
    if args.dry_run:
        say("done", f"dry run: nothing written [{time.time() - t_start:.0f} s]")
        return
    marker_ok = os.path.isfile(os.path.join(out, MARKER))
    if mode == "nothing" and marker_ok and not staged:
        say("done", f"nothing to do [{time.time() - t_start:.0f} s]")
        return

    # build (for mode=nothing only the tidying of a run that stopped late: staging and unlisted page files; nothing is read)
    os.makedirs(out, exist_ok=True)
    lock = acquire_lock(out)
    try:
        if mode != "nothing" and marker_ok:
            os.remove(os.path.join(out, MARKER))  # the collection is about to change: until this run finishes it is not a finished one
        say("build", f"{mode} started; pages {args.max_page_size_mb:g} MB, {args.read_threads} read threads")
        t0 = time.time()
        # keys: the complete matches to add (extend) or to build over (fresh / resume); the header's own matches stay whatever keys says
        result = curator.build_basic_tomes(
            channels, tome_name=TOME_NAME, header_tome_name=HEADER, keys=buildable,
            behavior_if_complete="extend", resume=True, max_page_size_mb=args.max_page_size_mb, read_threads=args.read_threads,
        )
        say("build", f"finished in {time.time() - t0:.0f} s")

        # verify: the collection is whole and lists what it should
        header_keys = result.header.get_keyset()
        problems = []
        if not result.header.is_complete:
            problems.append("the header is not complete")
        lacking = [k for k in buildable if k not in set(header_keys)]
        if lacking:
            problems.append(f"{len(lacking)} built matches are missing from the header, e.g. {lacking[0]}")
        total = 0
        for channel, loader in result.tomes.items():
            keys = loader.get_keyset()
            total += len(keys)
            if not loader.is_complete:
                problems.append(f"{channel}: tome not complete")
            elif not set(keys) <= set(header_keys):
                problems.append(f"{channel}: {len(set(keys) - set(header_keys))} matches not in the header")
        if problems:
            say("verify", "PROBLEMS: " + "; ".join(problems))
            sys.exit(1)
        say("verify", f"ok: header {len(header_keys)} matches, {len(result.tomes)} channel tomes ({total} match-channel pairs), all complete")
        write_atomically(record_path, json.dumps({"channels": sorted(known_channels | set(channels)), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        write_atomically(os.path.join(out, MARKER),
                         f"mode={mode} matches_in_header={len(header_keys)} matches_added={len(buildable)} finished={time.strftime('%Y-%m-%dT%H:%M:%S')}")
        if args.publish_chunks:
            publish_chunk(out, base, names, channels)
        say("done", f"{mode}: {len(buildable)} matches added, {len(header_keys)} in the collection, {time.time() - t_start:.0f} s in all")
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


if __name__ == "__main__":
    main()
