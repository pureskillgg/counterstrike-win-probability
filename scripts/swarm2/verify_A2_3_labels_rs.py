"""Side check for finding A2-3 (verifier): which channel agrees with round_state freeze-end labels?

For each 2024-02-g match, compares the round_freeze_end label set in round_state (the live-start
source used by cross_period.py) with the round_end label set and the round_start label set, grouped by
the label pattern found in verify_A2_3_labels.py. Prints to stdout only.

ASSUMPTIONS
  A1  Same reader as verify_A2_3_indep.py (all dataframe_* pages of the channel folder).
  A2  Round label = int(round). Only event_type == "round_freeze_end" rows of round_state are used.
  A3  Pattern groups are taken from the label check: RE/RST contiguous (normal) vs RE min = RST min + 1
      (shifted). Matches with no round_start rows are reported separately.
"""

import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def read(ch, cols):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    assert files, ch
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def labels(df):
    out = collections.defaultdict(set)
    for m, r in df.select([pl.col("match_key"), pl.col("round").cast(pl.Int64)]).unique().iter_rows():
        out[m].add(r)
    return out


def main():
    re_l = labels(read("round_end", ["match_key", "round"]))
    rst_l = labels(read("round_start", ["match_key", "round"]))
    rs = read("round_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type") == "round_freeze_end")
    fz_l = labels(rs)
    print(f"matches with round_freeze_end rows: {len(fz_l)}")

    groups = collections.Counter()
    verdicts = collections.Counter()
    for m, a in re_l.items():
        b = rst_l.get(m)
        if not b:
            g = "no round_start rows"
        elif min(a) == min(b) + 1:
            g = "shifted (RE min = RST min + 1)"
        elif min(a) == min(b):
            g = "normal (RE min = RST min)"
        else:
            g = "other"
        f = fz_l.get(m, set())
        if f == a:
            v = "freeze_end == RE labels"
        elif f == (b or set()):
            v = "freeze_end == RST labels"
        elif f == a | (b or set()):
            v = "freeze_end == RE u RST"
        else:
            v = "freeze_end differs from both (RE only: {} | freeze_end only: {})".format(
                len(a - f), len(f - a))
        groups[g] += 1
        verdicts[(g, v if not v.startswith("freeze_end differs") else "freeze_end differs from both")] += 1
    for (g, v), n in sorted(verdicts.items()):
        print(f"  {n:>6}  {g:<30} {v}")
    print(f"matches by group: {dict(groups)}")

    # for the shifted group, show one example of each label set
    for m, a in re_l.items():
        b = rst_l.get(m)
        if b and min(a) == min(b) + 1:
            print(f"example shifted match {m}:")
            print(f"  RE labels         {min(a)}..{max(a)} ({len(a)})")
            print(f"  RST labels        {min(b)}..{max(b)} ({len(b)})")
            f = fz_l.get(m, set())
            print(f"  freeze_end labels {min(f) if f else None}..{max(f) if f else None} ({len(f)})")
            break


if __name__ == "__main__":
    main()
