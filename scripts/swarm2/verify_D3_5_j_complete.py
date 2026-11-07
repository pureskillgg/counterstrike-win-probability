"""D3-5 verifier, step J: how the header-complete subset is defined, and round-13 inversion by completeness (2024 G:)."""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


H = rd("header", ["key", "t_starters_score_final", "ct_starters_score_final"])
hdr = {k: (t, c) for k, t, c in H.iter_rows()}
RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"])
RE = (RE.with_columns((pl.col("player_count") == 20).cast(pl.Int8).alias("p20"))
      .sort(["match_key", "round", "p20"], descending=[False, False, True])
      .unique(["match_key", "round"], keep="first"))
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
sec = RS.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("h"))
hmap = {m: h for m, h in sec.iter_rows()}
W = collections.defaultdict(dict)
for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    W[m][r] = w
F = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    F[m][r] = (t, c)
day = lambda m: int(m.split("/")[3])

diff = collections.Counter()
cls = collections.Counter()
inv13 = collections.Counter()
for m, wins in W.items():
    if m not in hdr or hdr[m][0] is None:
        continue
    ht, hc = hdr[m]
    rounds = sorted(wins)
    N = len(rounds)
    contig = rounds == list(range(1, N + 1))
    g = "12-25" if day(m) >= 12 else "<12"
    if not contig:
        cls[(g, "noncontig")] += 1
        continue
    d = N - (ht + hc)
    diff[(g, d if abs(d) <= 3 else ("big+" if d > 0 else "big-"))] += 1
    if g == "12-25" and hmap.get(m) == 12 and 13 in F[m] and 14 in F[m]:
        t1, c1 = F[m][13]
        t2, c2 = F[m][14]
        D = T if (t2 - t1, c2 - c1) == (1, 0) else (CT if (t2 - t1, c2 - c1) == (0, 1) else None)
        if D is not None:
            inv13[("complete" if d == 0 else "not_complete", "W!=D" if wins[13] != D else "W=D")] += 1
print("contiguous day groups: N - (hdrT + hdrC) distribution (count):")
for k in sorted(diff, key=lambda k: (k[0], str(k[1]))):
    print("  ", k, diff[k])
print("non-contiguous:", dict(cls))
print("round-13 winner vs delta, h12, by completeness:")
for k in sorted(inv13):
    print("  ", k, inv13[k])
