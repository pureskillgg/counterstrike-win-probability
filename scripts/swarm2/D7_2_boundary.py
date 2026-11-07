"""D7-2: exact per-match onset boundary of the round-13 mismatch (time = header match_date, UTC).
Reads the per-round table written by D7_1_onset.py (own scratch output in logs\swarm2).
ASSUMPTION: player_count == 20 (2024 convention, as in e2).
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
R = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_all.parquet"))
O = R.filter(pl.col("bomb_side").is_not_null() & (pl.col("player_count") == 20))

def boundary(df, label):
    d = df.sort("mdt")
    y = d["mm"].cast(pl.Int8).to_list()
    n = len(y)
    # errors if predict mismatch for index >= cut (single-step boundary, cut in 0..n)
    pre_mm = [0]*(n+1); pre_clean = [0]*(n+1)
    for i, v in enumerate(y):
        pre_mm[i+1] = pre_mm[i] + v
        pre_clean[i+1] = pre_clean[i] + (1 - v)
    best = None
    for c in range(n+1):
        err = pre_mm[c] + (pre_clean[n] - pre_clean[c])
        if best is None or err < best[0]:
            best = (err, c)
    err, c = best
    last_clean = d.filter(pl.col("mm") == False)["mdt"].max()
    first_mm = d.filter(pl.col("mm") == True)["mdt"].min()
    print(f"[{label}] n={n} mismatches={sum(y)} | best single-step boundary errors={err} "
          f"(cut before index {c}); last clean before cut={d[c-1,'mdt'] if c>0 else None}; "
          f"first mismatch at/after cut={d[c,'mdt'] if c<n else None}")
    print(f"[{label}] overall last clean={last_clean} | first mismatch={first_mm}")
    return d, c

r13 = O.filter(pl.col("round") == 13)
print("r13 rows:", r13.height, "unique match_key:", r13["match_key"].n_unique())
d13, c13 = boundary(r13, "r13")

print("\n-- r13 rows in window 2024-02-06 12:00 .. 2024-02-08 00:00 (UTC match_date), sorted --")
w = d13.filter((pl.col("mdt") >= pl.datetime(2024,2,6,12)) & (pl.col("mdt") < pl.datetime(2024,2,8,0)))
print(w.select(["mdt", "match_key", "map_name", "mm", "winner_team_code", "bomb_side", "planter_side", "tick", "out_tick"]).with_columns(pl.col("match_key").str.slice(0,40)))

print("\n-- r13 rows strictly before 2024-02-07 00:00 with mismatch (any):", 
      r13.filter((pl.col("mdt") < pl.datetime(2024,2,7)) & pl.col("mm")).height)
print("-- r13 rows on/after 2024-02-07 00:00 that are clean:",
      r13.filter((pl.col("mdt") >= pl.datetime(2024,2,7)) & (~pl.col("mm"))).height)
print("-- r13 rows on/after 2024-02-08 00:00 that are clean:",
      r13.filter((pl.col("mdt") >= pl.datetime(2024,2,8)) & (~pl.col("mm"))).height)

print("\n-- day-by-day r13 totals by match_date (UTC), 2024-02-05..2024-02-09 --")
print(r13.filter((pl.col("mdt") >= pl.datetime(2024,2,5)) & (pl.col("mdt") < pl.datetime(2024,2,10)))
      .with_columns(pl.col("mdt").dt.strftime("%m-%d %H").alias("hr"))
      .group_by("hr").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("hr"))

print("\n-- hour-of-day of the first mismatches on 2024-02-07 (UTC) --")
w7 = r13.filter((pl.col("mdt") >= pl.datetime(2024,2,7)) & (pl.col("mdt") < pl.datetime(2024,2,8)))
print(w7.select(["mdt","mm","map_name"]).sort("mdt"))

print("\n-- Non-r13 / non-r25 mismatches (other): round numbers and dates --")
oth = O.filter(pl.col("rbin") == "other")
print("other rounds with mismatch by round number:")
print(oth.filter(pl.col("mm")).group_by("round").len().sort("round").rows())
print("other mismatches by md_day:", oth.filter(pl.col("mm")).group_by("md_day").len().sort("md_day").rows())
print("other mismatches, earliest and latest mdt:", oth.filter(pl.col("mm"))["mdt"].min(), oth.filter(pl.col("mm"))["mdt"].max())
print("other mismatches before 2024-02-07:", oth.filter(pl.col("mm") & (pl.col("mdt") < pl.datetime(2024,2,7))).height)

print("\n-- r25 threshold check --")
r25 = O.filter(pl.col("round") == 25)
d25, c25 = boundary(r25, "r25")
print("r25 rows before 2024-02-07:", r25.filter(pl.col("mdt") < pl.datetime(2024,2,7)).height,
      "| r25 mismatches before 2024-02-07:", r25.filter((pl.col("mdt") < pl.datetime(2024,2,7)) & pl.col("mm")).height)

print("\n-- per-match: does a match's r13 mismatch imply the same match's r25 mismatch? --")
both = (r13.select(["match_key", "mm"]).rename({"mm": "mm13"})
        .join(r25.select(["match_key", "mm"]).rename({"mm": "mm25"}), on="match_key"))
print(both.group_by(["mm13", "mm25"]).len().sort(["mm13", "mm25"]).rows())

print("\n-- max round per match (MR12 vs MR15 check), for matches with r13 outcome --")
mr = R.group_by("match_key").agg(pl.col("round").max().alias("maxr"), pl.col("mdt").first())
print("max round distribution (all matches):", mr.group_by("maxr").len().sort("maxr").rows())
