"""D3-5 verifier, step F: engine halftime markers in 2024 G: tome (round_state phase and last-round-of-half announce).

Per match: the round number where round_announce_last_round_half occurs, the round where phase first becomes
'second' (halftime -> second), and the round where the score-field arrangement flips (unit-delta check at freeze rows).
Tabulates by day group.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
day = lambda m: int(m.split("/")[3])
grp = lambda m: "12-25" if day(m) >= 12 else "<12"

ann = (RS.filter(pl.col("event_type") == "round_announce_last_round_half")
       .group_by("match_key").agg(pl.col("round").min().alias("ann_round"), pl.len().alias("n_ann")))
print("matches with last_round_half announce:", ann.height, "| n_ann distribution:", ann.group_by("n_ann").len().rows())
print("ann_round distribution (all matches with the event):",
      sorted(collections.Counter(ann["ann_round"].to_list()).items())[:30])

# phase transitions: first round index where phase == 'second'
ph = (RS.filter(pl.col("phase").is_in(["first", "second", "halftime"]))
      .group_by(["match_key", "round", "phase"]).agg(pl.col("tick").min().alias("tk")))
first_second = (ph.filter(pl.col("phase") == "second").group_by("match_key")
                .agg(pl.col("round").min().alias("second_round")))
first_halftime = (ph.filter(pl.col("phase") == "halftime").group_by("match_key")
                  .agg(pl.col("round").min().alias("halftime_round")))
print("matches with phase second:", first_second.height, "| with halftime phase:", first_halftime.height)
print("second_round distribution:", sorted(collections.Counter(first_second["second_round"].to_list()).items())[:40])
print("halftime_round distribution:", sorted(collections.Counter(first_halftime["halftime_round"].to_list()).items())[:40])

# arrangement flips from freeze fields: freeze rows, unit delta check
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
flips = collections.defaultdict(list)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).sort(["match_key", "round"]).iter_rows():
    pass
rows = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    rows[m][r] = (t, c)
unclear_at = collections.Counter()
for m, d in rows.items():
    for r in sorted(d):
        if r + 1 in d:
            t1, c1 = d[r]
            t2, c2 = d[r + 1]
            if (t2 - t1, c2 - c1) not in [(1, 0), (0, 1)]:
                unclear_at[(grp(m), r)] += 1
print("non-unit freeze deltas by (grp, round) [freeze r -> r+1]:", sorted(unclear_at.items())[:60])

hm = ann.join(first_second, on="match_key", how="full", coalesce=True).join(first_halftime, on="match_key", how="full", coalesce=True)
hm = hm.with_columns(pl.col("match_key").map_elements(grp, return_dtype=pl.Utf8).alias("g"))
print("\nannounce_round vs second_round (counts):")
print(hm.group_by(["g", "ann_round", "second_round"]).len().sort(["g", "len"], descending=[False, True]).head(30))
print("\nhalftime_round counts by grp:")
print(hm.group_by(["g", "halftime_round"]).len().sort(["g", "len"], descending=[False, True]).head(30))
