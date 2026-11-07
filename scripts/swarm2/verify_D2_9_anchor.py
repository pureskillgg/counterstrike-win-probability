"""verify D2-9, independent anchor: compare round_end (E) with bomb_exploded / bomb_defused round_state events,
which do not use the round_start channel. Reports (E - bomb) in seconds (tick / 64) by match day for 2024 and 2026.
If the 2024 round_end channel is shifted by ~7 s on some days, E - bomb will shift by the same amount there.
Lazy scans, projection only.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_anchor.py
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
        E = first_tick(read(label, "round_end", ["match_key", "round", "tick"]), "E")
        rst = read(label, "round_state", ["match_key", "round", "tick", "event_type"])
        ev_counts = rst.group_by("event_type").agg(pl.len().alias("n")).filter(
            pl.col("event_type").str.contains("bomb")).rows()
        print(f"\n== [{label}] bomb-type round_state event counts: {ev_counts}")
        for ev, nm in [("bomb_exploded", "X"), ("bomb_defused", "D")]:
            B = first_tick(rst.filter(pl.col("event_type") == ev).select(["match_key", "round", "tick"]), nm)
            j = E.join(B, on=["match_key", "round"], how="inner")
            j = j.with_columns(((pl.col("E") - pl.col(nm)) / TICK_RATE).alias("E_minus_B_s"),
                               pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).alias("day"))
            print(f"-- {label} {ev}: rounds with a {nm} event and round_end = {j.height:,}; "
                  f"E - bomb overall (s): median={j['E_minus_B_s'].median():.2f}  "
                  f"q05={j['E_minus_B_s'].quantile(0.05, interpolation='nearest'):.2f}  "
                  f"q95={j['E_minus_B_s'].quantile(0.95, interpolation='nearest'):.2f}")
            by = j.group_by("day").agg(pl.len().alias("n"),
                                       pl.col("E_minus_B_s").median().round(2).alias("med_E_minus_B_s"),
                                       (pl.col("E_minus_B_s") > 3).mean().round(3).alias("frac_gt3s")).sort("day")
            print(by)


if __name__ == "__main__":
    main()
