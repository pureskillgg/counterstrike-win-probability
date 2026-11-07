"""A3: how often the 2024 key-folder day (csds/2024/02/DD/...) differs from the header match_date.

Read-only. The round-13 analysis (scripts/swarm/e2_side_check_r13.py) groups rounds by the key-folder
day. This counts disagreement overall and among duplicate copies (exact groups, bakeoff rule).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def read(ch, cols):
    fd = BASE / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def main():
    H = {r["key"]: r["match_date"] for r in read("header", ["key", "match_date"]).iter_rows(named=True)}
    rows = []
    for k, md in H.items():
        kdate = "-".join(k.split("/")[1:4])  # csds/2024/02/DD/... -> 2024-02-DD
        mdate = md[:10]                       # full YYYY-MM-DD (dates span Dec 2023 .. Feb 2024)
        rows.append((k, kdate, mdate, kdate == mdate))
    n = len(rows)
    agree = sum(1 for r in rows if r[3])
    print(f"2024 matches: {n}; key-folder date equals match_date date: {agree} ({100 * agree / n:.1f}%)")
    dis = collections.Counter((r[1], r[2]) for r in rows if not r[3])
    print(f"disagreeing (key-folder date, match_date) pairs, top 10: {dis.most_common(10)}")
    by_key = collections.Counter(r[1] for r in rows if r[1].startswith("2024-02-1") or r[1].startswith("2024-02-2"))
    by_md = collections.Counter(r[2] for r in rows if r[2].startswith("2024-02-1") or r[2].startswith("2024-02-2"))
    print("matches per key-folder date, 2024-02-12..25:",
          {f"2024-02-{i:02d}": by_key.get(f"2024-02-{i:02d}", 0) for i in range(12, 26)})
    print("matches per match_date, 2024-02-12..25:",
          {f"2024-02-{i:02d}": by_md.get(f"2024-02-{i:02d}", 0) for i in range(12, 26)})
    print("match_date months:", dict(collections.Counter(r[2][:7] for r in rows)))


if __name__ == "__main__":
    main()
