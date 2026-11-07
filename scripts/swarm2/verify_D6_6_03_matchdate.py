"""verify D6-6, step 03: is the folder day (match_key 'csds/2024/02/<DD>') the same as match_date?
Also: where does build_num flip on the match_date axis, and what do the 2026 OT boundary labels look like.
Reads only the 2024 header (one parquet) and the 2026/2024 round_state label columns.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_03_matchdate.py
"""
import collections
import glob
import re
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
B26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"

H = pl.read_parquet(glob.glob(str(B24 / "header" / "dataframe_*"))[0],
                    columns=["key", "match_date", "build_num", "map_name", "server_name"])
H = H.with_columns(
    pl.col("key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("keyday"),
    pl.col("match_date").str.slice(0, 10).alias("mdate"),
    pl.col("match_date").str.slice(0, 10).str.slice(8, 2).alias("mday"),
    pl.col("match_date").str.slice(0, 7).alias("mmonth"),
)
print("== A. key folder day vs match_date (2024) ==")
print("match_date month counts:", H.group_by("mmonth").len().sort("mmonth").rows())
ct = H.group_by(["keyday", "mdate"]).len().sort(["keyday", "mdate"])
print("keyday x match_date (first 3 dates per keyday):")
for kd in sorted(H["keyday"].unique().to_list()):
    sub = ct.filter(pl.col("keyday") == kd)
    print(f"  keyday {kd}: n={sub['len'].sum()} dates(top 4)={sub.sort('len', descending=True).head(4).rows()}")
same = H.filter(pl.col("keyday") == pl.col("mday")).height
print(f"matches where keyday == match_date day: {same} of {H.height}")

print("\n== B. build_num on the match_date axis ==")
g = H.group_by("build_num").agg(pl.col("match_date").min().alias("min_md"), pl.col("match_date").max().alias("max_md"),
                                 pl.len().alias("n"))
print(g.sort("build_num"))
b9842_max = H.filter(pl.col("build_num") == 9842)["match_date"].max()
b9961_min = H.filter(pl.col("build_num") == 9961)["match_date"].min()
print("max match_date of build 9842:", b9842_max, "| min match_date of build 9961:", b9961_min)
srt = H.sort("match_date")
idx = [i for i, b in enumerate(srt["build_num"].to_list()) if b == 9961][0]
print("around first 9961 (time order):")
print(srt.slice(max(0, idx - 4), 10).select(["match_date", "build_num", "keyday", "map_name"]))
# out-of-order exceptions: any 9842 after first 9961 or any 9961 before last 9842
bl = srt["build_num"].to_list()
last_9842 = max(i for i, b in enumerate(bl) if b == 9842)
first_9961 = min(i for i, b in enumerate(bl) if b == 9961)
print(f"time-order exceptions: 9842 rows after first 9961: {sum(1 for b in bl[first_9961:] if b == 9842)}; "
      f"9961 rows before last 9842: {sum(1 for b in bl[:last_9842] if b == 9961)}")
exc = srt.slice(first_9961 - 1, 4).select(["match_date", "build_num", "keyday"]) if False else None
print("rows with build 9842 dated on/after the first 9961 date:")
print(H.filter((pl.col("build_num") == 9842) & (pl.col("match_date") >= b9961_min)).select(["match_date", "build_num", "keyday", "map_name", "server_name"]))

print("\n== C. build by keyday (2024) ==")
print(H.group_by(["keyday", "build_num"]).len().sort(["keyday", "build_num"]))

print("\n== D. 2026 round_state transition labels at overtime-boundary rounds (control, W1 unambiguous) ==")
RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type", "t_score", "ct_score"])
                for f in sorted(glob.glob(str(sorted(B26.glob("basic_round_state.*"))[0] / "dataframe_*")))],
               how="vertical_relaxed")
RS = RS.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first", maintain_order=True)
scm = collections.defaultdict(dict)
for m, r, t, c in RS.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm[m][r] = (t, c)


def label(t0, c0, t1, c1):
    no = (t1 - t0, c1 - c0) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and sw:
        return "both"
    if no:
        return "no"
    if sw:
        return "swap"
    return "none"


labc = collections.Counter()
for m, s in scm.items():
    for r in s:
        if r + 1 in s and r in (24, 26, 27, 28, 32, 33, 34, 38, 39, 40, 44, 45, 46):
            labc[(r, label(*s[r], *s[r + 1]))] += 1
for k in sorted(labc):
    print(f"  round {k[0]} -> {k[0] + 1}: label {k[1]:<5} n={labc[k]}")
print("note: 'both' or 'none' labels are unscorable under W1, so they are excluded from the 25,417 control count.")
