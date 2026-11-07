"""verify D2-9, which channel carries the 2024 day-13+ offset: round_end (E) or the round_state window events
(freeze_end F, buytime_ended B)? Tests that do not use round_end:
  T-A: B - F (ticks), a round_state-internal interval, value counts by day group in 2024 and 2026.
  T-B: E - last player_death tick in the round (ticks), by day group, to see whether E carries an extra 429-tick mode.
Day groups: 2024 days 01-07 (E == bomb, canonical), 2024 days 13-25 (affected), 2026 (canonical).
Lazy scans, projection only.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_which_channel.py
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


def top_values(df, col, label):
    vc = df.group_by(col).agg(pl.len().alias("n")).sort("n", descending=True).head(6)
    tot = df.height
    return [(int(v), int(n), round(n / tot * 100, 1)) for v, n in vc.rows()]


def group_of(day_expr):
    d = day_expr.cast(pl.Int64)
    return (pl.when((d >= 1) & (d <= 7)).then(pl.lit("2024 d01-07"))
            .when((d >= 13) & (d <= 25)).then(pl.lit("2024 d13-25"))
            .otherwise(pl.lit("other")).alias("grp"))


def main():
    print("polars", pl.__version__)
    for label in ["2024", "2026"]:
        rst = read(label, "round_state", ["match_key", "round", "tick", "event_type"])
        F = first_tick(rst.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]), "F")
        B = first_tick(rst.filter(pl.col("event_type") == "buytime_ended").select(["match_key", "round", "tick"]), "B")
        E = first_tick(read(label, "round_end", ["match_key", "round", "tick"]), "E")
        D = read(label, "player_death", ["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
            pl.col("tick").max().alias("last_death"))
        day = pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1)

        fb = F.join(B, on=["match_key", "round"], how="inner").with_columns(
            (pl.col("B") - pl.col("F")).alias("B_minus_F"), day.alias("day")).with_columns(group_of(pl.col("day")))
        print(f"\n== [{label}] T-A: buytime_ended - freeze_end (ticks), rounds={fb.height:,}")
        for grp in sorted(fb["grp"].unique().to_list()):
            sub = fb.filter(pl.col("grp") == grp)
            print(f"   {grp}: n={sub.height:,} top values (tick, count, pct):", top_values(sub, "B_minus_F", label))

        ed = E.join(D, on=["match_key", "round"], how="inner").with_columns(
            (pl.col("E") - pl.col("last_death")).alias("E_minus_lastdeath"), day.alias("day")).with_columns(group_of(pl.col("day")))
        print(f"-- [{label}] T-B: round_end - last player_death (ticks), rounds={ed.height:,}")
        for grp in sorted(ed["grp"].unique().to_list()):
            sub = ed.filter(pl.col("grp") == grp)
            print(f"   {grp}: n={sub.height:,} top values (tick, count, pct):", top_values(sub, "E_minus_lastdeath", label))
            hi = sub.filter(pl.col("E_minus_lastdeath") >= 400).height
            print(f"      share with E - last_death >= 400 ticks: {hi / sub.height * 100:.1f}% ({hi:,})")


if __name__ == "__main__":
    main()
