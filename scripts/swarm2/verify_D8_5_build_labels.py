"""verify_D8_5_build_labels: why build 9842 has no complete matches (D8-5 verifier).

For days 12-25, by build: the round label range in round_end (min, max), the number of rows per match against the
header total, and the share of matches whose round labels are exactly 1..n. Appends to logs\\swarm2\\verify_D8_5.log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_build_labels.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
LOG = ROOT / "logs" / "swarm2" / "verify_D8_5.log"
LOGF = open(LOG, "a", encoding="utf-8", newline="\n")


def say(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    LOGF.write(s + "\n")
    LOGF.flush()


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


H = read_dir(BASE / "header", ["key", "build_num", "t_starters_score_final", "ct_starters_score_final"])
HD = {r["key"]: r for r in H.iter_rows(named=True)}
RE = read_dir(BASE / f"basic_round_end{SUF}", ["match_key", "round", "winner_team_code"])
per = RE.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                                   pl.len().alias("rows"), pl.col("round").n_unique().alias("nuniq"))
say("\n=== verify_D8_5 BUILD LABEL CHECK (appended, verify_D8_5_build_labels.py) ===")
stats = collections.defaultdict(collections.Counter)
for r in per.iter_rows(named=True):
    h = HD.get(r["match_key"])
    if h is None:
        continue
    day = int(r["match_key"].split("/")[3])
    if not (12 <= day <= 25):
        continue
    b = h["build_num"]
    total = int(h["t_starters_score_final"]) + int(h["ct_starters_score_final"])
    s = stats[b]
    s["matches"] += 1
    s[f"rmin={r['rmin']}"] += 1
    s["rows==total"] += int(r["rows"] == total)
    s["rows-total<0"] += int(r["rows"] < total)
    s["labels 1..n"] += int(r["rmin"] == 1 and r["rmax"] == r["rows"] and r["nuniq"] == r["rows"])
    s["rows-total"] += 0
for b in sorted(stats, key=lambda z: str(z)):
    s = stats[b]
    labels = sorted([(k, v) for k, v in s.items() if k.startswith("rmin=")], key=lambda z: -z[1])
    say(f"build {b}: days 12-25 matches={s['matches']}; rows==header total: {s['rows==total']}; "
        f"rows<header total: {s['rows-total<0']}; labels exactly 1..n: {s['labels 1..n']}; round-label min: {labels[:4]}")
LOGF.close()
print("LABELS DONE")
