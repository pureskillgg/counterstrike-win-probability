"""D3 census: which 2024 G: tome matches have raw E: data in the tail copy, and which channels.

Read-only. For each header match in the G: 2024 tome (key csds/2024/MM/DD/<uuid>/csds), checks the
E: folder E:\\csds_final\\csds\\csds_tail_recovered_20261011\\csds\\2024\\MM\\DD\\<uuid>\\ and which of the
four channels used by the comparison (round_end, round_state, bomb_state, bomb_action) are present.
Also splits the result by the round-13 pools used in D3_compare_tome_vs_raw.py.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D3_census_e_coverage.py
"""
import collections
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
E_BASE = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds")
CHANNELS = ["round_end", "round_state", "bomb_state", "bomb_action"]


def g_read(ch, cols=None):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


H = g_read("header", ["key", "match_date", "match_id"])
print("header matches:", H.height, "unique keys:", H["key"].n_unique())
print("match_date sample:", H["match_date"].head(3).to_list())

RE = g_read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
BS = g_read("bomb_state", ["match_key", "round", "event_type"])
outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
flag = (RE.filter(pl.col("player_count") == 20).join(outcome, on=["match_key", "round"], how="inner")
        .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("side"),
                      pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
        .with_columns((pl.col("side") != pl.col("winner_team_code")).alias("mm")))
r13 = flag.filter(pl.col("round") == 13)
aff_set = set(r13.filter((pl.col("day") >= 12) & pl.col("mm"))["match_key"].to_list())
clean_set = set(r13.filter((pl.col("day") <= 7) & (~pl.col("mm")))["match_key"].to_list())
r13_out_late = set(r13.filter(pl.col("day") >= 12)["match_key"].to_list())
print(f"affected pool {len(aff_set)}; clean pool {len(clean_set)}; late round-13 outcome {len(r13_out_late)}")

status = {}
rows = []
for key in H["key"].to_list():
    parts = key.split("/")
    p = E_BASE / parts[1] / parts[2] / parts[3] / parts[4]
    if not p.exists():
        st = "dir_missing"
        present = []
    else:
        names = set(os.listdir(p))
        present = [c for c in CHANNELS if c in names]
        if not names:
            st = "dir_empty"
        elif len(present) == len(CHANNELS):
            st = "all4"
        elif present:
            st = f"partial({len(present)}/4)"
        else:
            st = "no_compare_channels"
    status[key] = st
    rows.append((key, parts[3], st))

cnt = collections.Counter(st for _, _, st in rows)
print("\nE: coverage over all header matches:")
for st, n in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"  {st:<22} {n}")

by_day = collections.defaultdict(collections.Counter)
for key, day, st in rows:
    by_day[day][st] += 1
print("\nE: coverage by day (day: status counts):")
for day in sorted(by_day):
    print(f"  {day}: " + ", ".join(f"{k}={v}" for k, v in sorted(by_day[day].items())))


def pool_report(name, s):
    c = collections.Counter(status[k] for k in s)
    print(f"{name}: n={len(s)}; " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))


print("\nRound-13 pools vs E: coverage:")
pool_report("affected pool (day 12..25, r13 mismatch)", aff_set)
pool_report("clean pool (day 01..07, r13 outcome, no mm)", clean_set)
pool_report("late r13 outcome (day 12..25, all)", r13_out_late)

usable_aff = sorted(k for k in aff_set if status.get(k) == "all4")
usable_clean = sorted(k for k in clean_set if status.get(k) == "all4")
print(f"\nusable (all4) affected={len(usable_aff)} clean={len(usable_clean)}")
with open(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D3_usable_keys.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(["AFF " + k for k in usable_aff] + ["CLEAN " + k for k in usable_clean]) + "\n")
print("usable key list written to scratchpad D3_usable_keys.txt")
