"""Add new tome sources to an EXISTING combined store without rebuilding it (dynamic expansion).

  python -I -X utf8 scripts\\data\\append_combined.py [--out tomes\\combined] [--sources name1,name2,...] [--dry-run] [--no-backup]

build_combined.py rebuilds the whole store from every source. This script reads tomes/combined/COMBINED_BUILD.json (the ledger: which
sources are merged and how many pages each channel has), finds the sources that are not merged yet, and for every channel
  * checks that the new pages can be cast to the channel's EXISTING canonical schema without changing it: no new column, and the
    canonical dtype of every column computed over (existing type + new types) is unchanged;
  * writes the new rows as new pages dataframe_<next> (build_combined.convert_page: the same casts, BOOL_MAP and round_end.player_count_norm
    rule as the full build), rewrites keyset_00000 as the union of the keysets of ALL merged sources, and finally rewrites
    COMBINED_BUILD.json (temp file + fsync + os.replace: the commit point).
A channel whose schema WOULD change (a new column, or a canonical dtype change such as Null -> Float64) is not appended: it is rebuilt
from all sources with build_combined.build_channel() into <folder>.new and swapped in at the commit; the other channels are appended.
A channel the store does not have yet is written fresh.

Safety:
  * Recovery first, on every run (also when there is nothing new): a leftover <folder>.old means an earlier schema-change swap did not reach
    its commit, so the old folder is put back; a leftover <folder>.new, a channel folder the ledger does not know, pages beyond the ledger's
    page count and *.tmp files are removed. The store is then exactly the last committed state.
  * A copy of the committed store is kept as <out>.bak (built as <out>.bak.tmp and swapped in, so the previous .bak is never deleted before
    the new one is complete).
  * A channel the ledger lists but whose folder has no pages is a damaged store: the run stops instead of treating it as new.
  * Before the first write, every merged source needed for a schema-change rebuild is checked to still exist.
  * After the writes every touched channel is verified: rows on disk == ledger rows, keyset size == ledger keys.
test_append_equivalence.py checks that the result equals a full rebuild, page by page and row hash by row hash.
"""
import argparse
import collections
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
import pyarrow.parquet as pq

import build_combined as B

sys.stdout.reconfigure(encoding="utf-8")


def channel_folder(base, ch):
    return os.path.join(base, "header" if ch == "header" else f"basic_{ch}{B.SUFFIX}")


def pages_of(folder):
    return sorted(glob.glob(os.path.join(folder, "dataframe_*")))


def convert(df, ch, order, canon_t, fname):
    return B.convert_page(df, ch, order, canon_t, fname)


def plan_channel(ch, new_sources, per, base, ledger):
    """('skip'|'append'|'fresh'|'rebuild', detail) for one channel."""
    new_types, new_order = collections.defaultdict(set), []
    for name, _ in new_sources:
        for f in per[name].get(ch, {}).get("pages", []):
            sch = pq.read_schema(f)
            if len(sch.names) == 0:
                continue
            for n, t in zip(sch.names, sch.types):
                if n not in new_types:
                    new_order.append(n)
                new_types[n].add(str(t))
    folder = channel_folder(base, ch)
    in_ledger = ch in ledger["channels"]
    if in_ledger and not pages_of(folder):
        raise SystemExit(f"the ledger lists channel {ch!r} but {folder} has no pages: the store is damaged; restore it from the .bak copy or rebuild it with "
                         f"build_combined.py (not guessing: treating it as new would drop the rows of the merged sources)")
    if not new_order and not in_ledger:
        return "skip", "no columns in the new sources and none in the store"
    if not in_ledger:
        return "fresh", (new_order, new_types)
    # An existing channel falls through even when the new sources have no columns for it: they add no rows, but a full rebuild still records
    # them (zero rows) and unions their keysets, so the append does the same.
    first = pq.read_schema(pages_of(folder)[0])
    old_order = [n for n in first.names]
    old_arrow = {n: str(t) for n, t in zip(first.names, first.types)}
    old_canon = {n: B.canon({old_arrow[n]}) for n in old_order if n != "player_count_norm"}
    for n in new_order:
        if n not in old_arrow:
            return "rebuild", f"new column {n!r} in the new sources (old pages lack it)"
        if n == "player_count_norm":
            continue
        cu = B.canon({old_arrow[n]} | new_types[n])
        if cu != old_canon[n]:
            return "rebuild", f"column {n!r}: canonical dtype would change {old_canon[n]} -> {cu} (new types {sorted(new_types[n])})"
    return "append", (old_order, old_arrow, old_canon)


def full_build(out, want):
    return subprocess.call([sys.executable, "-I", "-X", "utf8", os.path.join(os.path.dirname(__file__), "build_combined.py"), "--out", out, "--sources", ",".join(want)])


def recover(base, ledger):
    """Bring the store back to the last committed state (see the module docstring)."""
    known = {channel_folder(base, ch) for ch in ledger["channels"]}
    for ch in ledger["channels"]:
        f = channel_folder(base, ch)
        if os.path.exists(f + ".old"):
            if os.path.exists(f):
                shutil.rmtree(f)
            os.replace(f + ".old", f)
            print(f"recovered {os.path.basename(f)}: an interrupted schema-change swap was rolled back")
        if os.path.exists(f + ".new"):
            shutil.rmtree(f + ".new")
            print(f"removed uncommitted rebuild {os.path.basename(f)}.new")
        info = ledger["channels"][ch]
        for p in pages_of(f):
            if int(os.path.basename(p).split("_")[1]) >= info["pages"]:
                os.remove(p)
                print(f"removed orphan page {os.path.relpath(p, base)}")
        for t in glob.glob(os.path.join(f, "*.tmp")):
            os.remove(t)
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if os.path.isdir(p) and p not in known and not p.endswith((".old", ".new")) and (d == "header" or d.endswith(B.SUFFIX)):
            shutil.rmtree(p)
            print(f"removed channel folder {d}: the ledger does not know it (an interrupted run wrote it)")


def source_keys(ledger_sources, ch):
    """Keys of channel `ch` according to the merged sources themselves (authoritative); None when a merged source no longer exists."""
    keys = set()
    for n, _ in ledger_sources:
        if not os.path.isdir(os.path.join(B.TOMES, n, "tome")):
            return None
        for k in B.list_channels(n).get(ch, {}).get("keysets", []):
            keys.update(pl.read_parquet(k)["_"].to_list())
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(B.TOMES, "combined"))
    ap.add_argument("--sources", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    args.out = os.path.normpath(args.out)
    want = [s for s in args.sources.split(",") if s] or [n for n, _ in B.DEFAULT_SOURCES]
    man_path = os.path.join(args.out, "COMBINED_BUILD.json")
    if not os.path.exists(man_path):
        if args.dry_run:
            print(f"{args.out} has no COMBINED_BUILD.json: a real run would do the full build of {want}; dry run: nothing written")
            return
        print(f"{args.out} has no COMBINED_BUILD.json: nothing to append to; running the full build")
        sys.exit(full_build(args.out, want))
    ledger = json.load(open(man_path, encoding="utf-8"))
    base = os.path.join(args.out, "tome", "csds")
    if not args.dry_run:
        recover(base, ledger)
    sources = B.resolve_sources(want)
    merged = {n for n, _ in ledger["sources"]}
    new_sources = [(n, e) for n, e in sources if n not in merged]
    print(f"merged sources: {len(merged)}; resolved from the request: {len(sources)}; new: {[n for n, _ in new_sources]}")
    if not new_sources:
        print("nothing new to add")
        return
    per = {n: B.list_channels(n) for n, _ in new_sources}
    channels = sorted(set(ledger["channels"]) | set().union(*[set(v) for v in per.values()]))
    plans, rebuild_reasons = {}, []
    for ch in channels:
        kind, detail = plan_channel(ch, new_sources, per, base, ledger)
        plans[ch] = (kind, detail)
        if kind == "rebuild":
            rebuild_reasons.append(f"{ch}: {detail}")
    print("plan:", dict(collections.Counter(k for k, _ in plans.values())))
    for r in rebuild_reasons:
        print("  schema change, this channel is rebuilt from all sources (the others are appended):", r)
    rebuild_chs = [ch for ch in channels if plans[ch][0] == "rebuild"]
    all_sources = [(n, e) for n, e in ledger["sources"]] + list(new_sources)
    if rebuild_chs:  # checked BEFORE anything is written: a missing merged source would otherwise leave the store half updated
        for n, _ in all_sources:
            if not os.path.isdir(os.path.join(B.TOMES, n, "tome")):
                raise SystemExit(f"cannot rebuild {rebuild_chs}: merged source {n} no longer exists under {B.TOMES}; restore it or run build_combined.py with the sources you have")
    if args.dry_run:
        return
    t0 = time.time()
    if not args.no_backup:
        bak, bak_tmp = args.out + ".bak", args.out + ".bak.tmp"
        if os.path.exists(bak_tmp):
            shutil.rmtree(bak_tmp)
        shutil.copytree(args.out, bak_tmp)  # the committed store (recovery ran above); the previous .bak stays until this copy is complete
        if os.path.exists(bak):
            shutil.rmtree(bak)
        os.replace(bak_tmp, bak)
        print(f"backup -> {bak} ({time.time() - t0:.0f}s)")
    new_names = [n for n, _ in new_sources]
    touched, n_fresh = {}, 0
    for ch in channels:
        kind, detail = plans[ch]
        if kind in ("skip", "rebuild"):
            continue
        folder = channel_folder(base, ch)
        info = ledger["channels"].get(ch)
        if kind == "fresh":
            n_fresh += 1
            order, types = detail
            canon_t = {n: B.canon(types[n]) for n in order}
            os.makedirs(folder, exist_ok=True)
            # the sources already merged had no columns for this channel (else it would exist), but a full rebuild still lists them with
            # zero rows and unions their keysets (the matches they processed): do the same
            info = {"pages": 0, "rows_by_source": {n: 0 for n, _ in ledger["sources"]}, "keys": 0, "columns": len(order) + (1 if ch == "round_end" and "player_count_norm" not in order else 0),
                    "cast_columns": sorted(n for n in order if len(types[n]) > 1)}
        else:
            old_order, old_arrow, old_canon = detail
            order = [n for n in old_order if n != "player_count_norm"]
            canon_t = dict(old_canon)
        keys = source_keys(ledger["sources"], ch)
        if keys is None:  # a merged source is gone: fall back to the keyset file of the store (committed state, thanks to recovery)
            print(f"warning: a merged source no longer exists; the keys of {ch} come from the store's keyset file")
            ks = os.path.join(folder, "keyset_00000")
            keys = set(pl.read_parquet(ks)["_"].to_list()) if os.path.exists(ks) else set()
        pages_before = info["pages"]
        page = pages_before
        added = 0
        for name in new_names:
            info["rows_by_source"][name] = 0
            for f in per[name].get(ch, {}).get("pages", []):
                if len(pq.read_schema(f).names) == 0:
                    continue
                df = pl.read_parquet(f)
                out = convert(df, ch, order, canon_t, f)
                out.write_parquet(os.path.join(folder, f"dataframe_{page:05d}"), compression="zstd")
                page += 1
                info["rows_by_source"][name] += df.height
                added += df.height
            for k in per[name].get(ch, {}).get("keysets", []):
                keys.update(pl.read_parquet(k)["_"].to_list())
        tmp_key = os.path.join(folder, "keyset_00000.tmp")
        pl.DataFrame({"_": sorted(keys)}).write_parquet(tmp_key, compression="zstd")
        os.replace(tmp_key, os.path.join(folder, "keyset_00000"))
        info["pages"], info["keys"] = page, len(keys)
        info["columns"] = len(pq.read_schema(pages_of(folder)[0]).names)  # the same definition as the full build: columns actually written
        touched[ch] = info
        print(f"{ch:<18} +{page - pages_before:>3} pages  +{added:>10} rows  keys={len(keys):>6}  ({kind})", flush=True)
    # channels whose schema changed: rebuilt from ALL sources (already merged + new) into <folder>.new, swapped in at the commit
    rebuilt = {}
    if rebuild_chs:
        per_all = {n: B.list_channels(n) for n, _ in all_sources}
        for ch in rebuild_chs:
            newf = channel_folder(base, ch) + ".new"
            if os.path.exists(newf):
                shutil.rmtree(newf)
            info = B.build_channel(ch, all_sources, per_all, newf)
            if info is not None:
                rebuilt[ch] = info
    # verify before committing the ledger
    for ch, info in rebuilt.items():
        newf = channel_folder(base, ch) + ".new"
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in pages_of(newf))
        if rows != sum(info["rows_by_source"].values()) or pl.read_parquet(os.path.join(newf, "keyset_00000")).height != info["keys"]:
            raise SystemExit(f"verification failed for rebuilt channel {ch}; the ledger was NOT updated (the next run rolls the store back; {args.out}.bak is the pre-append copy)")
    for ch, info in touched.items():
        folder = channel_folder(base, ch)
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in pages_of(folder))
        want_rows = sum(info["rows_by_source"].values())
        nkeys = pl.read_parquet(os.path.join(folder, "keyset_00000")).height
        if rows != want_rows or nkeys != info["keys"] or len(pages_of(folder)) != info["pages"]:
            raise SystemExit(f"verification failed for {ch}: rows on disk {rows} vs ledger {want_rows}, keys {nkeys} vs {info['keys']}, pages {len(pages_of(folder))} vs {info['pages']}; "
                             f"the ledger was NOT updated (the next run rolls the store back; {args.out}.bak is the pre-append copy)")
    for ch, info in touched.items():
        ledger["channels"][ch] = info
    # the swap of a rebuilt channel is two renames; recover() rolls an unfinished swap back (a leftover .old means 'not committed')
    old_folders = []
    for ch, info in rebuilt.items():
        folder = channel_folder(base, ch)
        os.replace(folder, folder + ".old")
        old_folders.append(folder + ".old")
        os.replace(folder + ".new", folder)
        ledger["channels"][ch] = info
    ledger["sources"] = list(ledger["sources"]) + [[n, e] for n, e in new_sources]
    ledger["appended"] = ledger.get("appended", []) + [{"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": new_names}]
    tmp = man_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, man_path)  # COMMIT
    for f in old_folders:
        shutil.rmtree(f, ignore_errors=True)
    print(f"appended {len(new_names)} source(s) to {args.out} in {time.time() - t0:.0f}s "
          f"({len(touched) - n_fresh} channels appended to, {n_fresh} written fresh, {len(rebuilt)} rebuilt for a schema change); ledger updated")


if __name__ == "__main__":
    main()
