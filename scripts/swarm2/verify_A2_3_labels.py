"""Side check for finding A2-3 (verifier): per-match label patterns of round_end vs round_start.

Question: the 4,463 direct misses are 4,435 final rounds plus 28 rounds of one match. Are the 1,957
round_start pairs with no same-label round_end row a label shift (round_end labels = round_start
labels + 1 in some matches), which would make the same-label join the wrong test for those matches?
Prints to stdout only. Reads tomes/2024-02-g and tomes/2026-10-01_to_08 with projected columns.

ASSUMPTIONS
  A1  Same reader as verify_A2_3_indep.py (all dataframe_* pages of the channel folder).
  A2  Round label = int(round). Pattern = (min, max) of the round labels per match.
"""

import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024-02-g": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}


def read(tome, suf, ch):
    files = sorted(glob.glob(str(tome / f"basic_{ch}{suf}" / "dataframe_*")))
    assert files, ch
    return pl.concat([pl.read_parquet(f, columns=["match_key", "round"]) for f in files], how="vertical_relaxed")


def labels(df):
    out = collections.defaultdict(set)
    for m, r in df.select([pl.col("match_key"), pl.col("round").cast(pl.Int64)]).unique().iter_rows():
        out[m].add(r)
    return out


def main():
    for name, (tome, suf) in TOMES.items():
        re_l = labels(read(tome, suf, "round_end"))
        rst_l = labels(read(tome, suf, "round_start"))
        print(f"== {name}: matches with round_end {len(re_l)}, with round_start {len(rst_l)}")
        pat = collections.Counter()
        ex = {}
        for m in re_l:
            a = re_l[m]
            b = rst_l.get(m)
            if not b:
                key = ("no round_start rows", min(a), max(a))
                pat["no round_start rows"] += 1
                ex.setdefault("no round_start rows", m)
                continue
            if a == set(range(min(a), max(a) + 1)) and b == set(range(min(b), max(b) + 1)):
                if min(a) == min(b) and max(b) == max(a) - 1:
                    p = "RE contiguous from {}, RST contiguous from {}, RST max = RE max - 1".format(min(a), min(b))
                elif min(a) == min(b) and max(b) == max(a):
                    p = "RE and RST identical contiguous ranges"
                elif min(a) == min(b) + 1 and max(b) == max(a) - 1:
                    p = "RE min = RST min + 1 and RST max = RE max - 1 (shifted by one)"
                elif min(a) == min(b) + 1 and max(b) == max(a) - 2:
                    p = "other shift (RE min = RST min + 1, RST max = RE max - 2)"
                else:
                    p = f"other contiguous (RE {min(a)}..{max(a)}, RST {min(b)}..{max(b)})"
            else:
                p = "non-contiguous label set (RE or RST has a gap)"
            pat[p] += 1
            ex.setdefault(p, (m, sorted(a)[:4] + ["..."] + sorted(a)[-3:], sorted(b)[:4] + ["..."] + sorted(b)[-3:]))
        total = sum(pat.values())
        for p, n in pat.most_common():
            print(f"  {n:>6}  {p}")
            if p in ex and n:
                print(f"          example: {ex[p]}")
        print(f"  total matches classified: {total}")

        # gaps in round_end label sets, and round_start labels that have no round_end label
        gap_m = sum(1 for m in re_l if re_l[m] != set(range(min(re_l[m]), max(re_l[m]) + 1)))
        ext_pairs = sum(len(rst_l.get(m, set()) - re_l[m]) for m in re_l)
        print(f"  round_end label sets with a gap: {gap_m}; round_start labels with no round_end label: {ext_pairs}")
        print()


if __name__ == "__main__":
    main()
