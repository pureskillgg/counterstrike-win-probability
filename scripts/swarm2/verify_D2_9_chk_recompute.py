"""verify_D2_9 independent recompute of the D2-9 claim.

Claim under test (D2_final.py section 6): for 2024, round_start tick > round_end tick in 79,063 of 79,063
rounds with both (median +20 ticks); for 2026, 0 of 35,992 (median -5,394). Proposed fix: window 2024
rounds from round_freeze_end to round_end, not round_start..round_end.

This script:
  A. Re-derives the section-6 numbers with the same reduction (sort by tick, unique keep=first per match, round)
     and with an alternative reduction (all rows, no dedup) to test whether dedup drives the result.
  B. Checks row multiplicity per (match, round) in round_start / round_end / round_freeze_end.
  C. Compares tick-based and second-based ordering of round_start vs round_end (second is a
     separate clock column in the same tome); tick/second slope per channel.
  D. Anchors each round with round_state event ticks (round_freeze_end, phase values) to see which
     of round_start or round_end is out of place, instead of assuming round_end is the anchor.
  E. Adjacent-round test: round_start(r) vs round_end(r-1) and round_end(r) vs round_start(r+1).

Lazy scans with column projection; one channel at a time. Output goes to stdout (captured into
logs\swarm2\verify_D2_9.log). Read-only on the tomes.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_chk_recompute.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def parts(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
    assert len(fd) == 1, (label, ch, fd)
    p = sorted(glob.glob(fd[0] + "/dataframe_*"))
    assert p, (label, ch)
    return p


def load(label, ch, cols):
    return pl.scan_parquet(parts(label, ch)).select(cols).collect()


def q(s, qs=(0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)):
    s = s.drop_nulls().sort()
    n = s.len()
    if n == 0:
        return "n/a"
    out = []
    for p in qs:
        i = min(n - 1, int(round(p * (n - 1))))
        out.append(f"{p:.2f}={s[i]}")
    return "  ".join(out)


def sign_counts(d):
    d = d.drop_nulls()
    return {"n": d.len(), "pos": int((d > 0).sum()), "zero": int((d == 0).sum()), "neg": int((d < 0).sum())}


results = {}

for label in ["2024", "2026"]:
    print(f"\n==================== {label} ====================")
    rs_raw = load(label, "round_start", ["match_key", "round", "tick", "second"])
    re_raw = load(label, "round_end", ["match_key", "round", "tick", "second", "winner_team_code"])
    print(f"rows: round_start={rs_raw.height:,} round_end={re_raw.height:,}")

    # ---- B. multiplicity
    for name, df in [("round_start", rs_raw), ("round_end", re_raw)]:
        g = df.group_by(["match_key", "round"]).len()
        print(f"[B] {name}: (match,round) keys={g.height:,} rows={df.height:,} "
              f"keys with >1 row={g.filter(pl.col('len') > 1).height:,} max rows per key={g['len'].max()}")
        print(f"    distinct matches={df['match_key'].n_unique():,}  round range={df['round'].min()}..{df['round'].max()}")
        print(f"    null tick={df['tick'].null_count()}  null second={df['second'].null_count()}")

    # ---- A. section-6 reduction (sort by tick, keep first per key), and alternative (no dedup)
    rs_d = rs_raw.sort("tick").unique(["match_key", "round"], keep="first")
    re_d = re_raw.sort("tick").unique(["match_key", "round"], keep="first")
    j = rs_d.join(re_d, on=["match_key", "round"], suffix="_end")
    d_tick = j.select((pl.col("tick") - pl.col("tick_end")).alias("d")).to_series()
    sc = sign_counts(d_tick)
    print(f"[A] section-6 reduction: rounds with both={j.height:,}  start>end={sc['pos']:,}  start==end={sc['zero']:,}  "
          f"start<end={sc['neg']:,}  median(start-end)={d_tick.median()}")
    print(f"    quantiles of (start-end) ticks: {q(d_tick)}")

    # alternative 1: first start in key vs LAST end in key (guards against duplicate end rows)
    rs_last = rs_raw.sort("tick").unique(["match_key", "round"], keep="last")
    re_last = re_raw.sort("tick").unique(["match_key", "round"], keep="last")
    j2 = rs_last.join(re_last, on=["match_key", "round"], suffix="_end")
    d2 = j2.select((pl.col("tick") - pl.col("tick_end")).alias("d")).to_series()
    s2 = sign_counts(d2)
    print(f"[A2] keep=last reduction: both={j2.height:,} start>end={s2['pos']:,} start<end={s2['neg']:,} "
          f"median={d2.median()}")

    # alternative 2: all rows pairwise by key (cartesian within key) -> min over start, max over end
    rs_min = rs_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("start_min"),
                                                         pl.col("tick").max().alias("start_max"))
    re_min = re_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("end_min"),
                                                         pl.col("tick").max().alias("end_max"))
    j3 = rs_min.join(re_min, on=["match_key", "round"])
    n_any_start_before_end = j3.filter(pl.col("start_min") < pl.col("end_max")).height
    n_all_start_after_end = j3.filter(pl.col("start_min") > pl.col("end_max")).height
    print(f"[A3] over all rows: keys with both={j3.height:,}  "
          f"min(start) < max(end) in {n_any_start_before_end:,}  min(start) > max(end) in {n_all_start_after_end:,}")

    # ---- C. tick vs second ordering
    d_sec = j.select((pl.col("second") - pl.col("second_end")).alias("d")).to_series()
    ss = sign_counts(d_sec)
    print(f"[C] second-based (start-end): start>end={ss['pos']:,} start==end={ss['zero']:,} start<end={ss['neg']:,} "
          f"median(start-end seconds)={d_sec.median()}")
    print(f"    quantiles of (start-end) seconds: {q(d_sec)}")
    agree = j.select(((pl.col("tick") - pl.col("tick_end")) * (pl.col("second") - pl.col("second_end")) > 0)
                     .alias("agree")).to_series()
    print(f"    tick and second agree on sign in {int(agree.sum()):,} of {agree.len():,} rounds")
    slope = j.filter((pl.col("second") - pl.col("second_end")).abs() > 1).select(
        ((pl.col("tick") - pl.col("tick_end")) / (pl.col("second") - pl.col("second_end"))).alias("tps")).to_series()
    print(f"    tick/second slope over rounds with |dsecond|>1s: n={slope.len():,} quantiles: {q(slope, (0.05, 0.25, 0.5, 0.75, 0.95))}")
    # per-row tick/second ratio on round_end (is tick a fixed clock?)
    ratio_end = re_raw.filter(pl.col("second") > 10).select((pl.col("tick") / pl.col("second")).alias("r")).to_series()
    ratio_start = rs_raw.filter(pl.col("second") > 10).select((pl.col("tick") / pl.col("second")).alias("r")).to_series()
    print(f"    tick/second ratio on round_end: {q(ratio_end, (0.05, 0.5, 0.95))}")
    print(f"    tick/second ratio on round_start: {q(ratio_start, (0.05, 0.5, 0.95))}")

    results[label] = {"join": j, "rs": rs_raw, "re": re_raw}

print("\n\n==================== round_state anchors ====================")
for label in ["2024", "2026"]:
    rst = load(label, "round_state", ["match_key", "round", "tick", "event_type", "phase", "second"])
    print(f"\n-- {label}: round_state rows={rst.height:,}")
    print("event_type counts:")
    print(rst.group_by("event_type").len().sort("len", descending=True).head(20))
    print("phase counts:")
    print(rst.group_by("phase").len().sort("len", descending=True).head(20))
