"""Follow-up 5 for D3-6 verifier. Read-only. Tests the bake-off live-start event for round 1.

bakeoff.py defines the live start as event_type == round_freeze_end in round_state (line ~237/570).
Check: for start-at-2 vs start-at-1 matches, is there a round_freeze_end row labelled round 1?
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

RE = g_read(T24 / f"basic_round_end{SUF24}", ["match_key", "round"])
first = RE.group_by("match_key").agg(pl.col("round").min().alias("fr"))
start2 = set(first.filter(pl.col("fr") == 2)["match_key"].to_list())
start1 = set(first.filter(pl.col("fr") == 1)["match_key"].to_list())
RS = g_read(T24 / f"basic_round_state{SUF24}", ["match_key", "round", "tick", "event_type"])
print("round_state columns used; event types:", sorted(RS["event_type"].unique().to_list())[:30])
fe = RS.filter(pl.col("event_type") == "round_freeze_end")
print("round_freeze_end rows:", fe.height)
for name, s in (("start2", start2), ("start1", start1)):
    sub = fe.filter(pl.col("match_key").is_in(list(s)))
    r1 = sub.filter(pl.col("round") == 1)
    m_r1 = r1["match_key"].n_unique()
    print(f"{name}: matches {len(s)}; freeze_end rows {sub.height}; matches with round-1 freeze_end {m_r1}")
    if name == "start2" and r1.height:
        print("  start2 round-1 freeze_end tick range:", r1["tick"].min(), r1["tick"].max())
    if name == "start1" and r1.height:
        print("  start1 round-1 freeze_end tick range:", r1["tick"].min(), r1["tick"].max())
print("DONE")
