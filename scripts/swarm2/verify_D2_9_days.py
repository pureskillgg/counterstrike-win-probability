"""verify D2-9, day-level breakdown: does the 2024 round_start / round_end / round_freeze_end / round_officially_ended
relation change regime by match day? Self-contained (same per-round timeline build as verify_D2_9_offsets.py).
Seconds = tick / 64 (tick rate measured by verify_D2_9_check.py S3). Lazy scans, projection only.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_days.py
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


def build(label):
    rst = read(label, "round_state", ["match_key", "round", "tick", "event_type"])
    parts_ = [first_tick(read(label, "round_start", ["match_key", "round", "tick"]), "S"),
              first_tick(read(label, "round_end", ["match_key", "round", "tick"]), "E")]
    for ev, nm in {"round_freeze_end": "F", "round_officially_ended": "O"}.items():
        sub = rst.filter(pl.col("event_type") == ev).select(["match_key", "round", "tick"])
        parts_.append(first_tick(sub, nm))
    t = parts_[0]
    for p in parts_[1:]:
        t = t.join(p, on=["match_key", "round"], how="full", coalesce=True)
    t = t.sort(["match_key", "round"])
    t = t.with_columns(pl.col("round").shift(-1).over("match_key").alias("nr"),
                       pl.col("F").shift(-1).over("match_key").alias("Fn"),
                       pl.col("S").shift(-1).over("match_key").alias("Sn"))
    t = t.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).alias("day"))
    return t


def day_table(label):
    t = build(label)
    j = t.filter(pl.col("S").is_not_null() & pl.col("E").is_not_null())
    nb = t.filter(pl.col("nr") == pl.col("round") + 1)
    out = j.with_columns(((pl.col("S") - pl.col("E")) / TICK_RATE).alias("S_minus_E_s"),
                         ((pl.col("O") - pl.col("E")) / TICK_RATE).alias("O_minus_E_s"),
                         ((pl.col("F") - pl.col("E")) / TICK_RATE).alias("F_minus_E_s"),
                         ((pl.col("S") - pl.col("F")) / TICK_RATE).alias("S_minus_F_s"))
    nbx = nb.filter(pl.col("S").is_not_null() & pl.col("Fn").is_not_null() & pl.col("E").is_not_null()).with_columns(
        ((pl.col("S") - pl.col("Fn")) / TICK_RATE).alias("S_minus_Fnext_s"),
        ((pl.col("Fn") - pl.col("E")) / TICK_RATE).alias("Fnext_minus_E_s"))
    a = out.group_by("day").agg(
        pl.len().alias("rounds_S_E"),
        (pl.col("S_minus_E_s") > 3).mean().round(3).alias("frac_S_minus_E_gt3s"),
        pl.col("S_minus_E_s").median().round(2).alias("med_S_minus_E_s"),
        pl.col("O_minus_E_s").median().round(2).alias("med_O_minus_E_s"),
        pl.col("F_minus_E_s").median().round(2).alias("med_window_F_to_E_s"),
        pl.col("S_minus_F_s").median().round(2).alias("med_S_minus_F_s"))
    b = nbx.group_by("day").agg(
        pl.len().alias("pairs_next"),
        pl.col("S_minus_Fnext_s").median().round(2).alias("med_S_minus_Fnext_s"),
        pl.col("Fnext_minus_E_s").median().round(2).alias("med_Fnext_minus_E_s"),
        pl.col("Fnext_minus_E_s").quantile(0.05, interpolation="nearest").round(2).alias("q05_Fnext_minus_E_s"),
        pl.col("Fnext_minus_E_s").quantile(0.95, interpolation="nearest").round(2).alias("q95_Fnext_minus_E_s"))
    return a.join(b, on="day", how="full", coalesce=True).sort("day")


def main():
    print("polars", pl.__version__)
    for label in ["2024", "2026"]:
        print(f"\n== [{label}] by match day: S=round_start, E=round_end, F=round_freeze_end, O=round_officially_ended")
        print(day_table(label))


if __name__ == "__main__":
    main()
