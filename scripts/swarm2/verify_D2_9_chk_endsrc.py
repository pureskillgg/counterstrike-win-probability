"""verify_D2_9 round_end source classification (strongest alternative to the proposed fix).

The 2024 round_start gap (start - end) is bimodal: about +20 ticks and about +448 ticks. Hypothesis to test:
for ~half of 2024 rounds the basic_round_end tick equals a reset-time event (cs_pre_restart) rather than the
round_end game event, so the proposed window end is late by ~7 s for those rounds. Test by classifying each
(match, round) key of basic_round_end by which round_state event shares its tick, then reporting, per class:
  S - E (start channel minus end channel), E - fe (live window length), and bomb-event containment.
Also reports the same classification for 2026 as a control (where basic_round_end == round_state round_end).

Read-only; lazy scans with projection; one channel at a time.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_chk_endsrc.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
TPS = 64.0  # ASSUMPTION: 64 ticks per second (tick/second ratio is exactly 64.0 in both tomes)


def parts(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
    assert len(fd) == 1, (label, ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def first_key(df, name):
    return (df.sort("tick").unique(["match_key", "round"], keep="first")
            .select(["match_key", "round", pl.col("tick").alias(name)]))


def desc(s):
    s = s.drop_nulls()
    if s.len() == 0:
        return "n=0"
    return (f"n={s.len():,} median={s.median()} p05={s.quantile(0.05)} p95={s.quantile(0.95)} "
            f"min={s.min()} max={s.max()}")


for label in ["2024", "2026"]:
    print(f"\n==================== {label} ====================")
    rst = pl.scan_parquet(parts(label, "round_state")).select(
        ["match_key", "round", "tick", "event_type"]).collect()
    E = first_key(pl.scan_parquet(parts(label, "round_end")).select(["match_key", "round", "tick"]).collect(), "E")
    S = first_key(pl.scan_parquet(parts(label, "round_start")).select(["match_key", "round", "tick"]).collect(), "S")
    fe = first_key(rst.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]), "fe")

    def ev(et):
        return first_key(rst.filter(pl.col("event_type") == et).select(["match_key", "round", "tick"]), et)

    base = E.join(S, on=["match_key", "round"], how="left").join(fe, on=["match_key", "round"], how="left")
    # ROUND_END candidates: exact-tick match to a round_state event on the same key
    for et in ["round_end", "cs_pre_restart", "cs_win_panel_round", "round_officially_ended", "round_prestart"]:
        base = base.join(ev(et), on=["match_key", "round"], how="left")
    base = base.with_columns(
        pl.when(pl.col("E") == pl.col("round_end")).then(pl.lit("A: =state round_end"))
        .when(pl.col("E") == pl.col("cs_pre_restart")).then(pl.lit("B: =state cs_pre_restart"))
        .when(pl.col("round_end").is_null() & pl.col("cs_pre_restart").is_null()).then(pl.lit("C: no matching event"))
        .otherwise(pl.lit("D: other")).alias("cls"))
    print(f"keys in basic_round_end: {base.height:,}")
    cls_counts = base.group_by("cls").len().sort("cls")
    print(cls_counts)

    base = base.with_columns(
        (pl.col("S") - pl.col("E")).alias("S_minus_E"),
        (pl.col("E") - pl.col("fe")).alias("E_minus_fe"),
    )
    for c in base["cls"].unique().sort().to_list():
        sub = base.filter(pl.col("cls") == c)
        print(f"\n  class {c}: keys={sub.height:,}")
        print("    S - E ticks:      " + desc(sub["S_minus_E"]))
        print("    E - fe seconds:   " + desc(sub["E_minus_fe"] / TPS))
        # bomb containment for keys in this class (bomb events from round_state)
        for et in ["bomb_defused", "bomb_exploded"]:
            b = ev(et).join(sub.select(["match_key", "round", "E", "fe"]), on=["match_key", "round"], how="inner")
            if b.height == 0:
                print(f"    {et}: no keys")
                continue
            after = b.filter(pl.col(et) > pl.col("E")).height
            print(f"    {et}: keys={b.height:,} after E={after:,} ({after / b.height:.4f})")
    # 2024: total keys where the round_end game event is absent from round_state
    if label == "2024":
        n_absent = base.filter(pl.col("round_end").is_null()).height
        print(f"\n  2024: keys in basic_round_end with NO round_state round_end event on same key: {n_absent:,} "
              f"of {base.height:,}")
