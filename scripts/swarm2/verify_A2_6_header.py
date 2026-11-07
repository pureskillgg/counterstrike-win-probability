# verify_A2_6 part 1 (header only, read-only).
# Recomputes: header match_date day vs folder day (2024 and 2026), full-date agreement,
# chance baseline, offset histogram, and a single-threshold build-clock test
# (does build_num separate cleanly on match_date or on folder day?).
# Reads only: tomes\2024-02-g\tome\csds\header\dataframe_00000 and
#             tomes\2026-10-01_to_08\tome\csds\header\dataframe_00000
import os
import sys
import re
import collections
import datetime as dt

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = r"G:\pureskill\csds-cs2"
H24 = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds", "header", "dataframe_00000")
H26 = os.path.join(BASE, "tomes", "2026-10-01_to_08", "tome", "csds", "header", "dataframe_00000")
COLS = ["key", "match_id", "match_date", "build_num", "map_name", "number_of_points",
        "t_starters_score_final", "ct_starters_score_final"]
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def load(path):
    df = pl.read_parquet(path, columns=COLS)
    parts = df["key"].str.split("/")
    df = df.with_columns(
        parts.list.len().alias("n_parts"),
        parts.list.get(0).alias("k_root"),
        parts.list.get(1).alias("k_year"),
        parts.list.get(2).alias("k_month"),
        parts.list.get(3).alias("k_day"),
        parts.list.get(4).alias("k_mid"),
        parts.list.get(5).alias("k_tail"),
        pl.col("match_date").str.slice(0, 10).alias("md_str"),
    )
    return df


def indep_chance(fd_counts, md_counts, n):
    # expected count of rows where folder day == match day-of-month if the two were independent
    return sum(fd_counts.get(d, 0) * md_counts.get(d, 0) for d in set(fd_counts) | set(md_counts)) / n


def threshold_errors(rows):
    # rows: list of (clock_value, y) with y = 1 if build is 9961. Best single cut, both orientations.
    rows = sorted(rows, key=lambda r: r[0])
    n1 = sum(y for _, y in rows)
    n0 = len(rows) - n1
    best = None
    left1 = left0 = 0
    i = 0
    # cut before each distinct clock value (and before the first / after the last)
    cuts = [0]
    for j in range(1, len(rows)):
        if rows[j][0] != rows[j - 1][0]:
            cuts.append(j)
    cuts.append(len(rows))
    for c in cuts:
        l1 = sum(y for _, y in rows[:c])
        l0 = c - l1
        r1 = n1 - l1
        r0 = n0 - l0
        e_a = l1 + r0          # low side = 9842, high side = 9961
        e_b = l0 + r1          # low side = 9961, high side = 9842
        for orient, e in (("low=9842,high=9961", e_a), ("low=9961,high=9842", e_b)):
            if best is None or e < best[0]:
                cut_val = rows[c][0] if c < len(rows) else None
                best = (e, orient, c, cut_val)
    return best, len(rows)


def report(label, df):
    print("=" * 100)
    print(label)
    n = df.height
    print("rows:", n, " distinct match_id:", df["match_id"].n_unique(), " distinct key:", df["key"].n_unique())
    print("key parts count:", dict(collections.Counter(df["n_parts"].to_list())),
          " root:", df["k_root"].unique().to_list(), " tail:", df["k_tail"].unique().to_list())
    print("folder year/month:", df.group_by(["k_year", "k_month"]).len().sort(["k_year", "k_month"]).rows())
    print("match_date pattern ok (YYYY-MM-DDTHH:MM:SS.mmmZ):",
          int(df["match_date"].str.contains(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$").sum()), "of", n)
    print("match_date sample:", df["match_date"].head(3).to_list())
    # day-only agreement (the claim's measure)
    day_eq = df.filter(pl.col("md_str").str.slice(8, 2) == pl.col("k_day"))
    print(f"[A] day-only agreement: header match_date day == folder day: {day_eq.height} of {n} = {day_eq.height / n:.4%}")
    # full-date agreement: folder date is 2024-02-DD / 2026-10-DD; compare YYYY-MM-DD strings
    full_eq = df.filter(pl.col("md_str") == (pl.col("k_year") + "-" + pl.col("k_month") + "-" + pl.col("k_day")))
    print(f"[B] full-date agreement (YYYY-MM-DD equal): {full_eq.height} of {n} = {full_eq.height / n:.4%}")
    if day_eq.height:
        print("    of the day-only agreements, match month distribution:",
              day_eq.with_columns(pl.col("md_str").str.slice(0, 7).alias("mon")).group_by("mon").len().sort("mon").rows())
    # chance baseline for [A]
    fd = collections.Counter(df["k_day"].to_list())
    md = collections.Counter(df["md_str"].str.slice(8, 2).to_list())
    print(f"[C] chance baseline for [A] under independence of folder day and match day-of-month: {indep_chance(fd, md, n):.1f} rows")
    # offset in days between folder date and match date (UTC date string)
    fdate = [dt.date(int(y), int(m), int(d)) for y, m, d in zip(df["k_year"], df["k_month"], df["k_day"])]
    mdate = [dt.date(int(s[0:4]), int(s[5:7]), int(s[8:10])) for s in df["md_str"]]
    diffs = [(f - m).days for f, m in zip(fdate, mdate)]
    buckets = collections.Counter()
    for x in diffs:
        if x <= -31:
            buckets["folder <= 31d BEFORE match"] += 1
        elif -30 <= x <= -2:
            buckets["folder 2-30d before match"] += 1
        elif x == -1:
            buckets["folder 1d before match"] += 1
        elif x == 0:
            buckets["same date"] += 1
        elif x == 1:
            buckets["folder 1d after match"] += 1
        elif 2 <= x <= 30:
            buckets["folder 2-30d after match"] += 1
        else:
            buckets["folder >= 31d after match"] += 1
    print("[D] folder date minus match date (UTC date) buckets:")
    for k in ["folder <= 31d BEFORE match", "folder 2-30d before match", "folder 1d before match", "same date",
              "folder 1d after match", "folder 2-30d after match", "folder >= 31d after match"]:
        print(f"      {k:28s} {buckets.get(k, 0):6d}  {buckets.get(k, 0) / n:7.2%}")
    print(f"    |offset| <= 1 day: {sum(1 for x in diffs if abs(x) <= 1)} of {n} = {sum(1 for x in diffs if abs(x) <= 1) / n:.2%}")
    print("    offset (days) quantiles p05/p25/p50/p75/p95:",
          [sorted(diffs)[int(q * (n - 1))] for q in (0.05, 0.25, 0.5, 0.75, 0.95)])
    print(f"    offset (folder - match, days): min {min(diffs)}, max {max(diffs)}; negative count {sum(1 for x in diffs if x < 0)}")
    # build_num by match date and by folder date
    print("[E] build_num by match_date month:", df.with_columns(pl.col("md_str").str.slice(0, 7).alias("mon"))
          .group_by(["mon", "build_num"]).len().sort(["mon", "build_num"]).rows())
    # build-clock test on 2024 only (builds are only 9842 and 9961 there)
    if set(df["build_num"].unique().to_list()) >= {9842, 9961} and df["build_num"].n_unique() == 2:
        rows_md = [(d, 1 if b == 9961 else 0) for d, b in zip(mdate, df["build_num"].to_list())]
        rows_fd = [(f, 1 if b == 9961 else 0) for f, b in zip(fdate, df["build_num"].to_list())]
        bm, nn = threshold_errors(rows_md)
        bf, _ = threshold_errors(rows_fd)
        print(f"[F] best single date cut separating build 9842 from 9961 (n={nn}):")
        print(f"      on match_date: errors={bm[0]} ({bm[0] / nn:.2%}) {bm[1]} cut_before={bm[3]}")
        print(f"      on folder date: errors={bf[0]} ({bf[0] / nn:.2%}) {bf[1]} cut_before={bf[3]}")
    print("[G] day-only agreement by folder day (first 2024 folder days if applicable):")
    tab = df.with_columns((pl.col("md_str").str.slice(8, 2) == pl.col("k_day")).alias("eq")) \
        .group_by("k_day").agg(pl.len().alias("n"), pl.col("eq").sum().alias("eq_day")).sort("k_day")
    print("      ", tab.rows())


h24 = load(H24)
report("2024-02-g header (4,435 matches expected)", h24)
h26 = load(H26)
report("2026-10-01_to_08 header (1,752 matches expected)", h26)
print("=" * 100)
print("DONE")
