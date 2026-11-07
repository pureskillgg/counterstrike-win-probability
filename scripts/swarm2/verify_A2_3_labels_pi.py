"""Side check for finding A2-3 (verifier): do player_info and player_disconnect labels follow round_end?

The 5v5 filter (player_info) and the disconnect filter run before the round_start check. If their round
labels were shifted like round_start in the 1,957 shifted matches, the removals of final rounds at those
stages would be label artifacts too. Prints to stdout only.

ASSUMPTIONS
  A1  Same reader as verify_A2_3_indep.py (all dataframe_* pages of the channel folder).
  A2  Round label = int(round). Groups: normal = RE min equals RST min; shifted = RE min = RST min + 1.
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
    pi_l = labels(read("player_info", ["match_key", "round"]))
    dc = read("player_disconnect", ["match_key", "round"])
    dc_pairs = collections.Counter()
    for m, r in dc.select([pl.col("match_key"), pl.col("round").cast(pl.Int64)]).iter_rows():
        dc_pairs[(m, r)] += 1

    def group(m):
        a = re_l[m]
        b = rst_l.get(m)
        if not b:
            return "no round_start rows"
        return "shifted" if min(a) == min(b) + 1 else "normal"

    print("== player_info label set vs round_end label set, per match group")
    rel = collections.Counter()
    for m in re_l:
        g = group(m)
        a = re_l[m]
        p = pi_l.get(m, set())
        if p == a:
            v = "PI == RE"
        elif p == rst_l.get(m, set()):
            v = "PI == RST"
        elif p == a - {max(a)}:
            v = "PI == RE minus final"
        elif p == {x - 1 for x in a}:
            v = "PI == RE shifted down by one"
        elif not p:
            v = "PI empty for match"
        else:
            v = "PI other"
        rel[(g, v)] += 1
    for (g, v), n in sorted(rel.items()):
        print(f"  {n:>6}  {g:<22} {v}")

    print("== player_info rows at final round labels (RE max), by group")
    fin = collections.Counter()
    for m in re_l:
        mx = max(re_l[m])
        fin[(group(m), mx in pi_l.get(m, set()))] += 1
    for (g, has), n in sorted(fin.items()):
        print(f"  {n:>6}  {g:<22} final round has any player_info row: {has}")

    print("== player_disconnect rows: label relative to round_end label set, by group")
    dcl = collections.Counter()
    for (m, r), n in dc_pairs.items():
        if m not in re_l:
            dcl[(group(m) if m in rst_l else "no round_start rows", "match not in round_end")] += n
            continue
        a = re_l[m]
        if r in a:
            v = "label in RE set"
        elif r in rst_l.get(m, set()):
            v = "label only in RST set"
        else:
            v = "label in neither"
        dcl[(group(m), v)] += n
    for (g, v), n in sorted(dcl.items()):
        print(f"  {n:>6}  {g:<22} {v}")


if __name__ == "__main__":
    main()
