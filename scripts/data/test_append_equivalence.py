"""Equivalence test: appending sources to a combined store must give the same store as rebuilding it from all sources.

  python -I -X utf8 scripts\\data\\test_append_equivalence.py --base-store tomes\\combined --extra 2026-10-09,2026-10-12_c01 [--work tomes\\_eq_test]

  1. FULL: build_combined.py over (the base store's sources + --extra) -> <work>\\full
  2. INC : copy of the base store (not modified) -> <work>\\inc, then append_combined.py --sources (base + extra) -> the extra sources are appended
  3. compare channel by channel, page by page: same channel set, same page count, same schema (names, order, dtypes), same row count,
     identical row hashes in identical order, identical key sets; and the ledgers (sources, rows_by_source, pages, keys, columns).
     `cast_columns` is informational only: an appended channel keeps the value of the first build.
Exit code 0 = equivalent, 1 = a difference (listed). The work folder is deleted at the start (only a folder whose name starts with '_eq').
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import append_combined as A
import build_combined as B

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(__file__)


def run(*cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.call([sys.executable, "-I", "-X", "utf8", *cmd])
    if r != 0:
        raise SystemExit(f"command failed with exit code {r}")


def compare(full, inc):
    diffs, info = [], []
    lf = json.load(open(os.path.join(full, "COMBINED_BUILD.json"), encoding="utf-8"))
    li = json.load(open(os.path.join(inc, "COMBINED_BUILD.json"), encoding="utf-8"))
    if [tuple(s) for s in lf["sources"]] != [tuple(s) for s in li["sources"]]:
        diffs.append(f"ledger sources differ: {lf['sources']} vs {li['sources']}")
    if set(lf["channels"]) != set(li["channels"]):
        diffs.append(f"channel sets differ: only full {sorted(set(lf['channels']) - set(li['channels']))}, only inc {sorted(set(li['channels']) - set(lf['channels']))}")
    bf, bi = os.path.join(full, "tome", "csds"), os.path.join(inc, "tome", "csds")
    n_pages = n_rows = 0
    for ch in sorted(set(lf["channels"]) & set(li["channels"])):
        a, b = lf["channels"][ch], li["channels"][ch]
        for k in ("pages", "keys", "columns", "rows_by_source"):
            if a[k] != b[k]:
                diffs.append(f"{ch}: ledger {k} differs: {a[k]} vs {b[k]}")
        if a["cast_columns"] != b["cast_columns"]:
            info.append(f"{ch}: cast_columns {a['cast_columns']} (full) vs {b['cast_columns']} (inc)")
        pf, pi = A.pages_of(A.channel_folder(bf, ch)), A.pages_of(A.channel_folder(bi, ch))
        if len(pf) != len(pi):
            diffs.append(f"{ch}: page count {len(pf)} vs {len(pi)}")
            continue
        kf = pl.read_parquet(os.path.join(A.channel_folder(bf, ch), "keyset_00000"))["_"].to_list()
        ki = pl.read_parquet(os.path.join(A.channel_folder(bi, ch), "keyset_00000"))["_"].to_list()
        if kf != ki:
            diffs.append(f"{ch}: key sets differ ({len(kf)} vs {len(ki)})")
        for x, y in zip(pf, pi):
            sx, sy = pq.read_schema(x), pq.read_schema(y)
            if sx.names != sy.names or [str(t) for t in sx.types] != [str(t) for t in sy.types]:
                diffs.append(f"{ch}/{os.path.basename(x)}: schema differs: {list(zip(sx.names, map(str, sx.types)))} vs {list(zip(sy.names, map(str, sy.types)))}")
                continue
            dx, dy = pl.read_parquet(x), pl.read_parquet(y)
            if dx.height != dy.height:
                diffs.append(f"{ch}/{os.path.basename(x)}: rows {dx.height} vs {dy.height}")
                continue
            if not np.array_equal(dx.hash_rows(seed=0).to_numpy(), dy.hash_rows(seed=0).to_numpy()):
                diffs.append(f"{ch}/{os.path.basename(x)}: row content or order differs")
            n_pages += 1
            n_rows += dx.height
        print(f"  {ch:<18} pages={len(pf):>3} compared", flush=True)
    return diffs, info, n_pages, n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-store", default=os.path.join(B.TOMES, "combined"))
    ap.add_argument("--extra", required=True, help="comma-separated sources to add on top of the base store's sources")
    ap.add_argument("--work", default=os.path.join(B.TOMES, "_eq_test"))
    ap.add_argument("--keep", action="store_true", help="keep the work folder")
    args = ap.parse_args()
    if not os.path.basename(args.work).startswith("_eq"):
        raise SystemExit("the work folder name must start with '_eq' (it is deleted)")
    base_led = json.load(open(os.path.join(args.base_store, "COMBINED_BUILD.json"), encoding="utf-8"))
    base_names = [n for n, _ in base_led["sources"]]
    extra = [s for s in args.extra.split(",") if s]
    want = base_names + [e for e in extra if e not in base_names]
    print(f"base store sources {base_names}\nextra {extra}", flush=True)
    if os.path.exists(args.work):
        shutil.rmtree(args.work)
    os.makedirs(args.work)
    full, inc = os.path.join(args.work, "full"), os.path.join(args.work, "inc")
    t0 = time.time()
    run(os.path.join(HERE, "build_combined.py"), "--out", full, "--sources", ",".join(want))
    print(f"full build done ({time.time() - t0:.0f}s)", flush=True)
    shutil.copytree(args.base_store, inc)
    t1 = time.time()
    run(os.path.join(HERE, "append_combined.py"), "--out", inc, "--sources", ",".join(want), "--no-backup")
    print(f"append done ({time.time() - t1:.0f}s; full build took {t1 - t0:.0f}s)", flush=True)
    # the test proves nothing if the append did nothing: every --extra source (or, for a day built in chunks, its chunks) must now be in BOTH ledgers
    for label, store in (("full build", full), ("appended store", inc)):
        merged = [s[0] for s in json.load(open(os.path.join(store, "COMBINED_BUILD.json"), encoding="utf-8"))["sources"]]
        missing = [e for e in extra if e not in merged and not any(m.startswith(e + "_c") for m in merged)]
        if missing:
            raise SystemExit(f"the {label} does not contain the extra source(s) {missing}: unresolvable name or no BUILD_OK; the comparison would be meaningless")
    diffs, info, n_pages, n_rows = compare(full, inc)
    for i in info:
        print("info:", i)
    if diffs:
        print(f"\nNOT EQUIVALENT: {len(diffs)} difference(s)")
        for d in diffs[:60]:
            print(" -", d)
        sys.exit(1)
    print(f"\nEQUIVALENT: {n_pages} pages, {n_rows:,} rows compared hash-for-hash, same order, same schemas, keys and ledgers")
    if not args.keep:
        shutil.rmtree(args.work)


if __name__ == "__main__":
    main()
