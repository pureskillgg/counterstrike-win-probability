"""verify D2-9, does the 2024 day-13+ offset sit in round_end or in the bomb events?
Test 1: bomb_exploded - bomb_planted (seconds) by day. The CS2 bomb timer is a fixed clock, so this interval should
        not depend on the day if the bomb events are correctly timed (ASSUMPTION: 40 s timer; reported, not assumed).
Test 2: distribution of (round_end - bomb) in ticks, to see whether the 2024 offset is one discrete value.
Test 3: per-match consistency of the offset in 2024 days 13-25 (all bomb rounds shifted, mixed, or none).
Lazy scans, projection only.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_bombplant.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
TICK_RATE = 64  # ASSUMPTION (measured: tick/second == 64 in every match)


def parts(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
    assert len(fd) == 1, (label, ch, fd)
    p = sorted(glob.glob(fd[0] + "/dataframe_*"))
    assert p, (label, ch)
    return p


def read(label, ch, cols):
    return pl.scan_parquet(parts(label, ch)).select(cols).collect()


def first_tick(df, name):
    d = df.sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True)
    return d.select(["match_key", "round", pl.col("tick").alias(name)])


def main():
    print("polars", pl.__version__)
    for label in ["2024", "2026"]:
        rst = read(label, "round_state", ["match_key", "round", "tick", "event_type"])
        P = first_tick(rst.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round", "tick"]), "P")
        X = first_tick(rst.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round", "tick"]), "X")
        E = first_tick(read(label, "round_end", ["match_key", "round", "tick"]), "E")
        day = pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1)

        print(f"\n== [{label}] Test 1: bomb_exploded - bomb_planted (s) by day")
        px = P.join(X, on=["match_key", "round"], how="inner").with_columns(
            ((pl.col("X") - pl.col("P")) / TICK_RATE).alias("X_minus_P_s"), day.alias("day"))
        g = px.group_by("day").agg(pl.len().alias("n"),
                                   pl.col("X_minus_P_s").median().round(2).alias("med_X_minus_P_s"),
                                   pl.col("X_minus_P_s").min().round(2).alias("min"),
                                   pl.col("X_minus_P_s").max().round(2).alias("max")).sort("day")
        print(g)

        print(f"-- Test 2: (round_end - bomb_exploded) in ticks, value counts (top 8) [{label}]")
        ex = E.join(X, on=["match_key", "round"], how="inner").with_columns(
            (pl.col("E") - pl.col("X")).alias("E_minus_X_ticks"))
        vc = ex.group_by("E_minus_X_ticks").agg(pl.len().alias("n")).sort("n", descending=True).head(8)
        print(vc.rows())

        if label == "2024":
            print("-- Test 3: per-match share of bomb-ended rounds with round_end - bomb_exploded > 3 s, days 13-25")
            ex2 = ex.with_columns(day.alias("day")).filter(pl.col("day").cast(pl.Int64) >= 13)
            ex2 = ex2.with_columns(((pl.col("E_minus_X_ticks") / TICK_RATE) > 3).alias("shift"))
            per_m = ex2.group_by("match_key").agg(pl.len().alias("n_bomb_rounds"),
                                                  pl.col("shift").mean().alias("share_shift"))
            print("   matches (days 13-25) with bomb rounds:", per_m.height,
                  "; all shifted:", per_m.filter(pl.col("share_shift") == 1.0).height,
                  "; none shifted:", per_m.filter(pl.col("share_shift") == 0.0).height,
                  "; mixed:", per_m.filter((pl.col("share_shift") > 0) & (pl.col("share_shift") < 1)).height)
            multi = per_m.filter(pl.col("n_bomb_rounds") >= 2)
            print("   matches with >=2 bomb rounds:", multi.height,
                  "; all shifted:", multi.filter(pl.col("share_shift") == 1.0).height,
                  "; none shifted:", multi.filter(pl.col("share_shift") == 0.0).height,
                  "; mixed:", multi.filter((pl.col("share_shift") > 0) & (pl.col("share_shift") < 1)).height)


if __name__ == "__main__":
    main()
