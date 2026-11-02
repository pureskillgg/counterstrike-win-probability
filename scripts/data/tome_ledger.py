"""Which matches of a day are already inside a finished tome? (the ledger run_tome_queue.ps1 uses to build only what is NEW)

  python -I -X utf8 scripts\\data\\tome_ledger.py --name 2026-10-11 [--tomes tomes]

Looks at the header keyset of every finished collection of that name: `tomes/<name>` (built whole) and `tomes/<name>_cNN` (chunks), a
collection being finished when it has a BUILD_OK marker. Prints one `dd/match-id` per line (the folder of the match under csds/yyyy/mm/),
then `#chunks <highest chunk number>` on the last line so the caller can number new chunks after the existing ones.
A header keyset is a one-column parquet of keys like `csds/2026/10/11/<id>/csds`.
"""
import argparse
import os
import re
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tomes", default=os.path.join(os.path.dirname(__file__), "..", "..", "tomes"))
    a = ap.parse_args()
    tomes = os.path.abspath(a.tomes)
    pat = re.compile(re.escape(a.name) + r"(?:_c(\d+))?$")
    seen, top = set(), 0
    for d in sorted(os.listdir(tomes)):
        m = pat.match(d)
        if not m or not os.path.isfile(os.path.join(tomes, d, "BUILD_OK")):
            continue
        ks = os.path.join(tomes, d, "tome", "csds", "header", "keyset_00000")
        if not os.path.isfile(ks):
            sys.exit(f"{d} has a BUILD_OK marker but no header keyset: refusing to guess what it holds")
        for k in pl.read_parquet(ks)["_"].to_list():
            parts = k.split("/")  # csds / yyyy / mm / dd / id / csds
            if len(parts) < 6:
                sys.exit(f"{d}: unexpected key {k!r}")
            seen.add(f"{parts[3]}/{parts[4]}")
        if m.group(1):
            top = max(top, int(m.group(1)))
    for s in sorted(seen):
        print(s)
    print(f"#chunks {top}")


if __name__ == "__main__":
    main()
