"""Side check for finding A2-3 (verifier): which label set is complete, by group?

Header final scores give the rounds played (t_starters_score_final + ct_starters_score_final), assuming a
completed match with no forfeit (ASSUMPTION H1). Compares that sum with the round_end count, the
round_start count and the round_end max label, per label group from verify_A2_3_labels.py.
Prints to stdout only.

ASSUMPTIONS
  H1  Header final score sum = number of rounds played (completed matches).
  H2  Same reader as verify_A2_3_indep.py.
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
    files = sorted(glob.glob(str(TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}") / "dataframe_*")))
    assert files, ch
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def labels(df):
    out = collections.defaultdict(set)
    for m, r in df.select([pl.col("match_key"), pl.col("round").cast(pl.Int64)]).unique().iter_rows():
        out[m].add(r)
    return out


def main():
    hdr = read("header", ["key", "t_starters_score_final", "ct_starters_score_final"])
    hsum = {k: (int(t or 0) + int(c or 0)) for k, t, c in hdr.iter_rows()}
    re_l = labels(read("round_end", ["match_key", "round"]))
    rst_l = labels(read("round_start", ["match_key", "round"]))
    tab = collections.Counter()
    for m, a in re_l.items():
        b = rst_l.get(m)
        if not b:
            g = "no round_start rows"
        else:
            g = "shifted" if min(a) == min(b) + 1 else "normal"
        s = hsum.get(m)
        if s is None:
            rel = "not in header"
        else:
            rel = (f"header sum = RE count" if s == len(a) else
                   f"header sum = RE count + 1" if s == len(a) + 1 else
                   f"header sum = RE count - 1" if s == len(a) - 1 else "header sum other")
            if b:
                rel += f"; header sum = RST count" if s == len(b) else (
                    f"; header sum = RST count + 1" if s == len(b) + 1 else "")
        tab[(g, rel)] += 1
    for (g, rel), n in sorted(tab.items()):
        print(f"  {n:>6}  {g:<22} {rel}")


if __name__ == "__main__":
    main()
