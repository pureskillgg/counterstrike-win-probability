"""Follow-up 2 for D3-6 verifier. Read-only. Appends to logs\\swarm2\\verify_D3_6.log.

  Q. map_name distribution across all G: 2024 header keys (any non-de_ maps?).
  R. Exact match_date duplicates in G: 2024 header; match of the 3 archive-only orphan headers.
  S. Round-1 timing: G: round_state round-1 tick vs first round_end tick, for start-2 matches
     and start-1 matches (days 12-25). ASSUMPTION: the first round_end row in G: marks the
     end of the first labelled round.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"


def g_read(folder, cols=None):
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def g24(ch, cols=None):
    return g_read(T24 / ("header" if ch == "header" else f"basic_{ch}{SUF24}"), cols)


def kday(k):
    return int(k.split("/")[3])


H = g24("header", ["key", "map_name", "match_date", "tick_rate", "build_num", "match_type"])
keys = H["key"].to_list()
print("== Q. map_name across all G: header keys")
print(dict(collections.Counter(H["map_name"].to_list())))
print("non-de_ maps:", sum(1 for m in H["map_name"].to_list() if not str(m).startswith("de_")))
print("tick_rate:", dict(collections.Counter(H["tick_rate"].to_list())))
print("match_type:", dict(collections.Counter(H["match_type"].to_list())))

print("\n== R. exact match_date duplicates in G: header")
md_counts = collections.Counter(H["match_date"].to_list())
dups = {m: n for m, n in md_counts.items() if n > 1}
print("match_date values shared by >1 key:", len(dups), "| keys involved:", sum(dups.values()))
orph_md = {
    "day06 0a04b57c": "2024-02-02T06:02:14.000Z",
    "day15 6c7bdef3": "2024-02-10T16:40:00.000Z",
    "day16 a295e0d7": "2024-02-16T04:34:25.000Z",
}
for name, md in orph_md.items():
    hits = [k for k, m in zip(keys, H["match_date"].to_list()) if m == md]
    print(f"  orphan {name} match_date {md}: G: keys with same date = {hits}")

print("\n== S. Round-1 timing (round_state round 1 vs first round_end)")
RE = g24("round_end", ["match_key", "round", "tick"])
first_end = RE.group_by("match_key").agg(pl.col("tick").min().alias("first_end_tick"),
                                         pl.col("round").min().alias("first_round"))
RS = g24("round_state", ["match_key", "round", "tick"])
rs1 = RS.filter(pl.col("round") == 1).select(["match_key", "tick"])
j = rs1.join(first_end, on="match_key", how="inner")
start_class = dict(zip(first_end["match_key"].to_list(), first_end["first_round"].to_list()))
print("rows (round_state round 1) with a first_end: ", j.height)
agg = (j.with_columns((pl.col("tick") <= pl.col("first_end_tick")).alias("before_first_end"))
         .group_by("match_key")
         .agg(pl.col("before_first_end").mean().alias("share_before"),
              pl.col("tick").max().alias("max_tick"),
              pl.col("tick").min().alias("min_tick"),
              pl.col("first_end_tick").first().alias("first_end_tick"),
              pl.len().alias("n_rs1")))
agg = agg.with_columns(pl.col("match_key").map_elements(lambda m: "start2" if start_class.get(m) == 2 else "start1",
                                                      return_dtype=pl.Utf8).alias("cls"))
for cls in ("start1", "start2"):
    a = agg.filter(pl.col("cls") == cls)
    if a.height == 0:
        continue
    print(f"  {cls}: matches {a.height}; n_rs1 rows median {a['n_rs1'].median()}; "
          f"share of round-1 state rows with tick <= first round_end tick: mean {a['share_before'].mean():.3f}; "
          f"matches fully before first end: {int((a['share_before'] == 1.0).sum())}")
    print(f"    round-1 state max tick median {a['max_tick'].median()}; first round_end tick median {a['first_end_tick'].median()}")
    print(f"    round-1 state min tick median {a['min_tick'].median()}")

# per-tick_rate sanity: tick_rate for start-2 vs start-1 (days 12-25)
tr = collections.defaultdict(collections.Counter)
late = [k for k in keys if kday(k) >= 12]
st2 = set(agg.filter(pl.col("cls") == "start2")["match_key"].to_list())
for k, tk in zip(H["key"].to_list(), H["tick_rate"].to_list()):
    if kday(k) >= 12:
        tr[tk]["start2" if k in st2 else "start1"] += 1
print("tick_rate by class (days 12-25):", {k: dict(v) for k, v in tr.items()})

print("\nDONE")
