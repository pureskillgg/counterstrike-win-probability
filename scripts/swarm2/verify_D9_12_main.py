"""D9-12 independent verifier, main analysis. Read-only on the tomes.

Recomputes the round_start label-offset claim (notes\\swarm2\\D9.md, D9-12) on BOTH tomes:
  1. Claim counts: rounds with a round_start row and a round_end row; start vs own end and next end.
  2. Within-channel anchor: round_state round_prestart / round_poststart vs round_freeze_end and round_end
     (same channel, same round counter as freeze_end).
  3. Anchor from bomb (planted / defused / exploded) and player_death ticks placed in each label's grid.
  4. Freeze-time spacing (freeze_end minus start) in ticks and seconds.
  5. Per-match label sets: round_start table vs round_end.

ASSUMPTIONS (marked):
  ASSUMPTION A: the grid of label r is (round_end(r-1), round_end(r)]; its live window is (round_freeze_end(r), round_end(r)].
  ASSUMPTION B: duplicate (match, round) keys collapse to the minimum tick; duplicates are counted, not dropped silently.
  ASSUMPTION C: seconds use the header tick_rate (read from the tome header, not assumed).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024-02-g": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026-10-01_to_08": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}
KEY = ["match_key", "round"]


def scan(base, suf, ch, cols):
    files = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    return pl.scan_parquet(files).select(cols).collect()


def mins(df, name):
    """Per (match, round): minimum tick, and the row count (to expose duplicates)."""
    return df.group_by(KEY).agg(pl.col("tick").min().alias(name), pl.len().alias(name + "_n"))


def bin_expr(t):
    """Place tick expression t in the grid of its own label (needs columns end_prev, fe, end, end_next)."""
    none = pl.lit(None, dtype=pl.String)
    return (pl.when(t.is_null()).then(none)
            .when(pl.col("end_prev").is_not_null() & (t <= pl.col("end_prev"))).then(pl.lit("A <=end(r-1)"))
            .when(t <= pl.col("fe")).then(pl.lit("B (end(r-1),fe(r)] freeze"))
            .when(t <= pl.col("end")).then(pl.lit("C live (fe(r),end(r)]"))
            .when(pl.col("end_next").is_not_null() & (t <= pl.col("end_next"))).then(pl.lit("D (end(r),end(r+1)]"))
            .when(pl.col("end_next").is_null()).then(pl.lit("X after last end"))
            .otherwise(pl.lit("E >end(r+1)")))


def quant(df, col):
    s = df[col].drop_nulls()
    if s.len() == 0:
        return "n=0"
    return (f"n={s.len()} min={s.min()} p01={s.quantile(0.01)} p05={s.quantile(0.05)} "
            f"p25={s.quantile(0.25)} p50={s.quantile(0.5)} p75={s.quantile(0.75)} "
            f"p95={s.quantile(0.95)} p99={s.quantile(0.99)} max={s.max()}")


def bins_str(df):
    rows = df.group_by("bin").len().sort("bin").rows()
    return " | ".join(f"{b}: {k}" for b, k in rows if b is not None)


for tome, (base, suf) in TOMES.items():
    print(f"\n########## tome {tome}")
    hdr = pl.concat([pl.read_parquet(f, columns=["tick_rate"]) for f in
                     sorted(glob.glob(str(base / "header" / "dataframe_*")))])
    print("header tick_rate value counts:", hdr["tick_rate"].value_counts().rows())

    RE_raw = scan(base, suf, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    RE = mins(RE_raw, "end")
    print(f"round_end: rows={RE_raw.height} unique(match,round)={RE.height} "
          f"dup keys={RE_raw.height - RE.height} matches={RE_raw['match_key'].n_unique()}")

    RS = scan(base, suf, "round_state", ["match_key", "round", "tick", "event_type"])
    FE = mins(RS.filter(pl.col("event_type") == "round_freeze_end"), "fe")
    PRE = mins(RS.filter(pl.col("event_type") == "round_prestart"), "pre")
    POST = mins(RS.filter(pl.col("event_type") == "round_poststart"), "post")
    RSS_STATE = mins(RS.filter(pl.col("event_type") == "round_start"), "rss_state")
    print(f"round_state keys: freeze_end={FE.height} (dup rows {int(FE['fe_n'].sum()) - FE.height}); "
          f"prestart={PRE.height}; poststart={POST.height}; state round_start={RSS_STATE.height}")

    RST_raw = scan(base, suf, "round_start", ["match_key", "round", "tick", "second"])
    RST = mins(RST_raw, "rst")
    print(f"round_start table: rows={RST_raw.height} unique(match,round)={RST.height} "
          f"matches={RST_raw['match_key'].n_unique()}")

    BS = scan(base, suf, "bomb_state", ["match_key", "round", "tick", "event_type"])
    PLANT = mins(BS.filter(pl.col("event_type") == "bomb_planted"), "plant")
    DEF = mins(BS.filter(pl.col("event_type") == "bomb_defused"), "defuse")
    EXP = mins(BS.filter(pl.col("event_type") == "bomb_exploded"), "explode")
    print(f"bomb_state keys: planted={PLANT.height} defused={DEF.height} exploded={EXP.height}")

    DTH = scan(base, suf, "player_death", ["match_key", "round", "tick"])
    print(f"player_death rows={DTH.height}")

    # Frame: one row per label key, with the neighbouring labels' ends and freeze ends.
    keys = pl.concat([RE.select(KEY), FE.select(KEY), RST.select(KEY)]).unique()
    E_prev = RE.select(KEY + ["end"]).with_columns((pl.col("round") + 1).alias("round")) \
        .rename({"end": "end_prev"})
    E_next = RE.select(KEY + ["end"]).with_columns((pl.col("round") - 1).alias("round")) \
        .rename({"end": "end_next"})
    F_next = FE.select(KEY + ["fe"]).with_columns((pl.col("round") - 1).alias("round")) \
        .rename({"fe": "fe_next"})
    FR = (keys.join(RE.select(KEY + ["end"]), on=KEY, how="left")
          .join(FE.select(KEY + ["fe"]), on=KEY, how="left")
          .join(E_prev, on=KEY, how="left")
          .join(E_next, on=KEY, how="left")
          .join(F_next, on=KEY, how="left"))
    print(f"label keys in union of round_end / freeze_end / round_start table: {FR.height}")

    print("\n  -- 1. Event family placed in the grid of its own label (A: <=end(r-1); B: freeze window; "
          "C: live; D: (end(r),end(r+1)]; E: later)")

    def classify(tbl, tcol, label):
        df = tbl.select(KEY + [tcol]).rename({tcol: "t"}).join(FR, on=KEY, how="left")
        df = df.with_columns(bin_expr(pl.col("t")).alias("bin"))
        print(f"   {label:30s} n={df.height:7d} -> {bins_str(df)}")
        return df

    classify(RST, "rst", "round_start table")
    classify(PRE, "pre", "state round_prestart")
    classify(POST, "post", "state round_poststart")
    classify(RSS_STATE, "rss_state", "state round_start")
    classify(PLANT, "plant", "bomb planted")
    classify(DEF, "defuse", "bomb defused")
    classify(EXP, "explode", "bomb exploded")
    dd = DTH.rename({"tick": "t"}).join(FR, on=KEY, how="left").with_columns(bin_expr(pl.col("t")).alias("bin"))
    print(f"   {'player_death (rows)':30s} n={dd.height:7d} -> {bins_str(dd)}")

    print("\n  -- 2. Claim replication (round_start table vs round_end of same / neighbouring label)")
    both = RST.join(FR, on=KEY, how="inner").filter(pl.col("end").is_not_null())
    print(f"   rounds with round_start row and round_end row: {both.height}")
    print(f"   start > own end: {both.filter(pl.col('rst') > pl.col('end')).height} | "
          f"start < own end: {both.filter(pl.col('rst') < pl.col('end')).height} | "
          f"start == own end: {both.filter(pl.col('rst') == pl.col('end')).height}")
    print(f"   start in (own end, next end]: "
          f"{both.filter((pl.col('rst') > pl.col('end')) & (pl.col('rst') <= pl.col('end_next'))).height} "
          f"(of {both.filter(pl.col('end_next').is_not_null()).height} with a next end)")
    print(f"   start in (prev end, own end]: "
          f"{both.filter((pl.col('rst') > pl.col('end_prev')) & (pl.col('rst') <= pl.col('end'))).height}")
    fe_both = FR.filter(pl.col("fe").is_not_null() & pl.col("end").is_not_null())
    print(f"   freeze_end rounds with own end: {fe_both.height}; freeze_end < own end: "
          f"{fe_both.filter(pl.col('fe') < pl.col('end')).height}; freeze_end > own end: "
          f"{fe_both.filter(pl.col('fe') > pl.col('end')).height}")
    fe_prev_ok = fe_both.filter(pl.col("end_prev").is_not_null())
    print(f"   freeze_end in (prev end, own end]: "
          f"{fe_prev_ok.filter((pl.col('fe') > pl.col('end_prev')) & (pl.col('fe') <= pl.col('end'))).height} "
          f"of {fe_prev_ok.height} with a previous end")

    print("\n  -- 3. Spacing (ticks; seconds = ticks / tick_rate)")
    tick_rate = int(hdr["tick_rate"].mode()[0])
    SP = FR.join(RST.select(KEY + ["rst"]), on=KEY, how="left").join(PRE.select(KEY + ["pre"]), on=KEY, how="left")
    SP = SP.with_columns(
        (pl.col("fe") - pl.col("rst")).alias("fe_minus_rst"),
        (pl.col("fe_next") - pl.col("rst")).alias("fe_next_minus_rst"),
        (pl.col("fe") - pl.col("pre")).alias("fe_minus_pre"),
        (pl.col("end") - pl.col("fe")).alias("live_len"),
        (pl.col("fe") - pl.col("end_prev")).alias("dead_time"))
    for c, lab in [("fe_minus_rst", "freeze_end(r) - round_start table(r)"),
                   ("fe_next_minus_rst", "freeze_end(r+1) - round_start table(r)"),
                   ("fe_minus_pre", "freeze_end(r) - state prestart(r)"),
                   ("live_len", "round_end(r) - freeze_end(r)"),
                   ("dead_time", "freeze_end(r) - round_end(r-1)")]:
        s = SP[c].drop_nulls()
        print(f"   {lab}: {quant(SP, c)}")
        if s.len() > 0:
            print(f"      as seconds at {tick_rate} ticks/s: p05={s.quantile(0.05) / tick_rate:.2f} "
                  f"p50={s.quantile(0.5) / tick_rate:.2f} p95={s.quantile(0.95) / tick_rate:.2f}")
    same = SP.filter(pl.col("rst").is_not_null() & pl.col("pre").is_not_null())
    print(f"   round_start table tick == state prestart tick (same label): "
          f"{same.filter(pl.col('rst') == pl.col('pre')).height} of {same.height}")
    tm = RST_raw.with_columns((pl.col("tick") - tick_rate * pl.col("second")).alias("d"))
    print(f"   round_start table tick - {tick_rate}*second: min={tm['d'].min()} max={tm['d'].max()} "
          f"abs>1 tick count={tm.filter(pl.col('d').abs() > 1).height}")

    print("\n  -- 4. Per-match label sets (round_start table minus round_end)")
    LAB_RE = RE.group_by("match_key").agg(pl.col("round").min().alias("re_min"),
                                          pl.col("round").max().alias("re_max"),
                                          pl.col("round").n_unique().alias("re_n"))
    LAB_RST = RST.group_by("match_key").agg(pl.col("round").min().alias("rst_min"),
                                            pl.col("round").max().alias("rst_max"),
                                            pl.col("round").n_unique().alias("rst_n"))
    LAB = LAB_RE.join(LAB_RST, on="match_key", how="full", coalesce=True)
    LAB = LAB.with_columns((pl.col("rst_min") - pl.col("re_min")).alias("d_min"),
                           (pl.col("rst_max") - pl.col("re_max")).alias("d_max"),
                           (pl.col("rst_n") - pl.col("re_n")).alias("d_n"))
    print("   (d_min, d_max, d_n) pattern counts, most common first:",
          LAB.group_by(["d_min", "d_max", "d_n"]).len().sort("len", descending=True).head(10).rows())
    print("   matches with round_end rows but no round_start rows:", LAB.filter(pl.col("rst_n").is_null()).height)
    print("   round_start labels equal to 0:", RST.filter(pl.col("round") == 0).height,
          "| round_end labels equal to 0:", RE.filter(pl.col("round") == 0).height)
    last_end = RE.join(LAB_RE, on="match_key").filter(pl.col("round") == pl.col("re_max"))
    print("   last round_end label per match lacking a round_start row:",
          last_end.join(RST, on=KEY, how="anti").height, "of", last_end.height)
    first_end = RE.filter(pl.col("round") == 1)
    print("   round-1 round_end rows lacking a round_start row:",
          first_end.join(RST, on=KEY, how="anti").height, "of", first_end.height)
