"""D3 secondary: provenance and round-numbering coverage of the 2024 G: tome against the E: tail copy.

1. For header matches whose E: folder is empty (day 01 mostly), does the G: tome hold round_end rows?
2. Round numbering: for each G: match, is the round set 1..n (no gap), missing only round 1 (start offset),
   or has interior gaps? Counts by day group.
3. player_count distribution by day group (the e2 side check keeps player_count == 20 only).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D3_coverage_gaps.py
"""
import collections
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
E_BASE = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds")


def g_read(ch, cols=None):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


H = g_read("header", ["key"])
RE = g_read("round_end", ["match_key", "round", "player_count"])
RE_u = RE.unique(["match_key", "round"], keep="first")
rounds_by_match = collections.defaultdict(list)
for m, r in RE_u.select(["match_key", "round"]).iter_rows():
    rounds_by_match[m].append(r)
pc_by_grp = collections.Counter()
for m, pc in RE.select(["match_key", "player_count"]).iter_rows():
    grp = "day12_25" if int(m.split("/")[3]) >= 12 else "day01_07"
    pc_by_grp[(grp, pc == 20)] += 1
print("player_count == 20 rows (grp, is20): count:", dict(pc_by_grp))

# round numbering
cls = collections.Counter()
cls_grp = collections.defaultdict(collections.Counter)
for m, rs in rounds_by_match.items():
    grp = "day12_25" if int(m.split("/")[3]) >= 12 else "day01_07"
    rs = sorted(rs)
    n = len(rs)
    if rs == list(range(1, n + 1)):
        c = "contiguous_from_1"
    elif rs == list(range(rs[0], rs[0] + n)) and rs[0] > 1:
        c = f"starts_at_{rs[0]}_contiguous"
    else:
        c = "interior_gap_or_other"
    cls[c] += 1
    cls_grp[grp][c] += 1
print("\nround numbering classes (all G: matches with round_end):")
for k, v in cls.most_common():
    print(f"  {k:<28} {v}")
for g in ["day01_07", "day12_25"]:
    print(f"  {g}: {dict(cls_grp[g])}")
starts = collections.Counter(min(rs) for rs in rounds_by_match.values())
print("first round value counts:", sorted(starts.items())[:8])

# provenance: E: empty folders vs G: rows
empty_or_missing = []
for key in H["key"].to_list():
    parts = key.split("/")
    if not parts[3] == "01":
        continue
    d = E_BASE / parts[1] / parts[2] / parts[3] / parts[4]
    names = set(os.listdir(d)) if d.exists() else set()
    if not any(c in names for c in ["round_end", "round_state", "bomb_state", "bomb_action"]):
        empty_or_missing.append(key)
g_rows = sum(1 for k in empty_or_missing if k in rounds_by_match)
g_round_rows = sum(len(rounds_by_match.get(k, [])) for k in empty_or_missing)
print(f"\nday-01 header matches with no compare channels in E: {len(empty_or_missing)}")
print(f"  of which have round_end rows in G: {g_rows}; total G round_end rows for them: {g_round_rows}")
print("  sample:", empty_or_missing[:2])
