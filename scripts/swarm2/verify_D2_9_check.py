"""verify D2-9: independent recomputation of the 2024 round_start vs round_end tick order, plus the
strongest alternative explanations and a check of the proposed round_freeze_end -> round_end window.

Lazy scans with column projection, one channel at a time (2024 round channels are < 1.1M rows).
Output goes to stdout; the run is captured into logs\\swarm2\\verify_D2_9.log.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_check.py
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
LABELS = ["2024", "2026"]
QS = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]


def parts(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
    assert len(fd) == 1, (label, ch, fd)
    p = sorted(glob.glob(fd[0] + "/dataframe_*"))
    assert p, (label, ch)
    return p


def read(label, ch, cols):
    return pl.scan_parquet(parts(label, ch)).select(cols).collect()


def dedup_first(df):
    # same selection rule as scripts/swarm2/D2_final.py section 6: earliest tick per (match_key, round)
    return df.sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True)


def qline(s):
    s = s.drop_nulls()
    if s.len() == 0:
        return "(empty)"
    return "  ".join(f"q{int(q * 100):03d}={s.quantile(q, interpolation='nearest'):,.1f}" for q in QS)


def s1_claim(label):
    print(f"\n== S1 [{label}] claim recomputation: rounds with both round_start and round_end")
    rs_raw = read(label, "round_start", ["match_key", "round", "tick", "second"])
    re_raw = read(label, "round_end", ["match_key", "round", "tick", "second"])
    print(f"   raw rows: round_start={rs_raw.height:,} round_end={re_raw.height:,}; "
          f"null tick: start={rs_raw['tick'].null_count()} end={re_raw['tick'].null_count()}; "
          f"null second: start={rs_raw['second'].null_count()} end={re_raw['second'].null_count()}")
    rs = dedup_first(rs_raw)
    re_ = dedup_first(re_raw)
    j = rs.join(re_, on=["match_key", "round"], how="inner", suffix="_end")
    d = j.with_columns((pl.col("tick") - pl.col("tick_end")).alias("dt"),
                       (pl.col("second") - pl.col("second_end")).alias("dsec"))
    n = d.height
    print(f"   rounds with both (earliest-tick rule): {n:,}")
    print(f"   tick: start > end {d.filter(pl.col('dt') > 0).height:,}; start == end "
          f"{d.filter(pl.col('dt') == 0).height:,}; start < end {d.filter(pl.col('dt') < 0).height:,}")
    print(f"   tick diff (start - end): {qline(d['dt'])}")
    print(f"   second: start > end {d.filter(pl.col('dsec') > 0).height:,}; start < end "
          f"{d.filter(pl.col('dsec') < 0).height:,}; null {d.filter(pl.col('dsec').is_null()).height:,}")
    print(f"   second diff (start - end, s): {qline(d['dsec'])}")
    return d


def s2_multiplicity(label):
    print(f"\n== S2 [{label}] event multiplicity per (match_key, round)")
    for ch in ["round_start", "round_end"]:
        df = read(label, ch, ["match_key", "round", "tick"])
        g = df.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("tick").n_unique().alias("nt"))
        dist = g.group_by("n").agg(pl.len().alias("keys")).sort("n")
        print(f"   {ch}: rows={df.height:,} matches={df['match_key'].n_unique():,} keys={g.height:,} "
              f"round range=[{df['round'].min()},{df['round'].max()}]")
        print("     rows-per-key distribution:", dist.rows())
        print(f"     keys with more than one distinct tick: {g.filter(pl.col('nt') > 1).height:,}")


def s3_second_vs_tick(label):
    print(f"\n== S3 [{label}] tick rate implied by (tick, second) within each match (round_state)")
    st = read(label, "round_state", ["match_key", "tick", "second"]).drop_nulls()
    g = st.group_by("match_key").agg((pl.col("tick").max() - pl.col("tick").min()).alias("dtk"),
                                     (pl.col("second").max() - pl.col("second").min()).alias("dsc"))
    g = g.filter(pl.col("dsc") > 60).with_columns((pl.col("dtk") / pl.col("dsc")).alias("tps"))
    print(f"   matches={g.height:,}; ticks per second: {qline(g['tps'])}")


def s4_freeze_window(label):
    print(f"\n== S4 [{label}] proposed window round_freeze_end -> round_end, and round_start placement")
    rst = read(label, "round_state", ["match_key", "round", "tick", "second", "event_type"])
    et = rst.group_by("event_type").agg(pl.len().alias("n")).sort("n", descending=True)
    print("   round_state event_type counts:", et.rows())
    fe_raw = rst.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick", "second"])
    g = fe_raw.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
    print(f"   round_freeze_end rows={fe_raw.height:,} keys={g.height:,} keys with >1 row={g.filter(pl.col('n') > 1).height:,}")
    fe = dedup_first(fe_raw)
    re_ = dedup_first(read(label, "round_end", ["match_key", "round", "tick", "second"]))
    rs = dedup_first(read(label, "round_start", ["match_key", "round", "tick", "second"]))
    fer = fe.join(re_, on=["match_key", "round"], how="inner", suffix="_end")
    fer = fer.with_columns((pl.col("tick") - pl.col("tick_end")).alias("dt"),
                           (pl.col("second") - pl.col("second_end")).alias("dsec"))
    print(f"   freeze_end joined to round_end: {fer.height:,} rounds")
    print(f"   freeze_end tick < round_end tick: {fer.filter(pl.col('dt') < 0).height:,}; "
          f"freeze_end tick > round_end tick: {fer.filter(pl.col('dt') > 0).height:,}")
    print(f"   freeze_end - round_end (ticks): {qline(fer['dt'])}")
    print(f"   round_end - freeze_end (s, from second): {qline(-fer['dsec'])}")
    fes = fe.join(rs, on=["match_key", "round"], how="inner", suffix="_start")
    fes = fes.with_columns((pl.col("tick_start") - pl.col("tick")).alias("dt_rs_fe"))
    print(f"   round_start joined to freeze_end: {fes.height:,}; round_start tick < freeze_end tick: "
          f"{fes.filter(pl.col('dt_rs_fe') < 0).height:,}; round_start tick > freeze_end tick: "
          f"{fes.filter(pl.col('dt_rs_fe') > 0).height:,}")
    print(f"   round_start - freeze_end (ticks): {qline(fes['dt_rs_fe'])}")
    # state-timeline cross-check of the channel ticks: round_state event rows named round_start / round_end
    evs = set(et["event_type"].to_list())
    for ev, ch_df, nm in [("round_start", rs, "round_start"), ("round_end", re_, "round_end")]:
        if ev not in evs:
            print(f"   round_state has no event_type '{ev}'; skipping timeline cross-check for {nm}")
            continue
        se = dedup_first(rst.filter(pl.col("event_type") == ev).select(["match_key", "round", "tick", "second"]))
        x = se.join(ch_df, on=["match_key", "round"], how="inner", suffix="_ch")
        x = x.with_columns((pl.col("tick") - pl.col("tick_ch")).alias("dt"))
        print(f"   round_state '{ev}' tick - channel {nm} tick: keys={x.height:,}; zero={x.filter(pl.col('dt') == 0).height:,}; "
              f"state > channel={x.filter(pl.col('dt') > 0).height:,}; state < channel={x.filter(pl.col('dt') < 0).height:,}; "
              f"{qline(x['dt'])}")


def s5_neighbour_shift(label):
    print(f"\n== S5 [{label}] round_start(r) vs round_end(r+k): ordering under shifted round numbers")
    rs = dedup_first(read(label, "round_start", ["match_key", "round", "tick"]))
    re_ = dedup_first(read(label, "round_end", ["match_key", "round", "tick"]))
    for k in [-1, 0, 1]:
        re_k = re_.with_columns((pl.col("round") - k).alias("round"))
        j = rs.join(re_k, on=["match_key", "round"], how="inner", suffix="_end")
        d = j.with_columns((pl.col("tick") - pl.col("tick_end")).alias("dt"))
        n = d.height
        pos = d.filter(pl.col("dt") > 0).height
        neg = d.filter(pl.col("dt") < 0).height
        print(f"   k={k:+d}: pairs={n:,}; start(r) > end(r{k:+d}): {pos:,} ({pos / n * 100:.2f}%); "
              f"start(r) < end(r{k:+d}): {neg:,} ({neg / n * 100:.2f}%); median dt={d['dt'].median()}")


def s6_monotone(label):
    print(f"\n== S6 [{label}] consecutive-round tick monotonicity within match")
    for ch in ["round_start", "round_end"]:
        df = dedup_first(read(label, ch, ["match_key", "round", "tick"])).sort(["match_key", "round"])
        nxt = df.with_columns(pl.col("tick").shift(-1).over("match_key").alias("nt"),
                              pl.col("round").shift(-1).over("match_key").alias("nr"))
        ok = nxt.filter(pl.col("nr") == pl.col("round") + 1)
        print(f"   {ch}: consecutive pairs={ok.height:,}; tick(r+1) > tick(r): "
              f"{ok.filter(pl.col('nt') > pl.col('tick')).height:,}; "
              f"tick(r+1) < tick(r): {ok.filter(pl.col('nt') < pl.col('tick')).height:,}")


def s7_by_day(label, d):
    print(f"\n== S7 [{label}] tick diff (start - end) by match day (from match_key path)")
    dd = d.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).alias("day"))
    agg = dd.group_by("day").agg(pl.len().alias("rounds"), (pl.col("dt") > 0).sum().alias("start_after_end"),
                                 pl.col("dt").median().alias("median_dt")).sort("day")
    print(agg)


def main():
    print("polars", pl.__version__)
    results = {}
    for label in LABELS:
        results[label] = s1_claim(label)
    for label in LABELS:
        s2_multiplicity(label)
    for label in LABELS:
        s3_second_vs_tick(label)
    for label in LABELS:
        s4_freeze_window(label)
    for label in LABELS:
        s5_neighbour_shift(label)
    for label in LABELS:
        s6_monotone(label)
    s7_by_day("2024", results["2024"])


if __name__ == "__main__":
    main()
