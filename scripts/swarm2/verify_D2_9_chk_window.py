"""verify_D2_9 window-endpoint check (proposed fix: 2024 window = round_freeze_end .. round_end).

Tests whether the endpoints of the proposed window are internally consistent and whether they agree with
round-scoped events that do not come from the basic_round_start channel:
  fe(r)  = first round_state tick with event_type == round_freeze_end for (match, round r)
  E(r)   = first basic_round_end tick for (match, round r)
  S(r)   = first basic_round_start tick for (match, round r)  (the channel under dispute)
  bomb events (round_state): bomb_planted, bomb_defused, bomb_exploded, first tick per (match, round)

Checks:
  1. E(r) - fe(r) distribution in seconds (64 ticks/s), both years. Sign and magnitude.
  2. Interval containment: fe(r) < bomb tick(r) <= E(r) for each bomb event type, both years.
  3. Previous-round containment for 2024: E(r-1) < fe(r) < E(r).
  4. round_state rows keyed to round r: min tick vs fe(r), max tick vs E(r).
  5. S(r) - fe(r) for comparison (2026 should be about -1280 ticks = freeze time).
Read-only; lazy scans with projection; one channel at a time.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_chk_window.py
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
TPS = 64.0  # ASSUMPTION: 64 ticks per second (verified: tick/second ratio is exactly 64.0 in both tomes)


def parts(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
    assert len(fd) == 1, (label, ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def first_key(df, name):
    return (df.sort("tick").unique(["match_key", "round"], keep="first")
            .select(["match_key", "round", pl.col("tick").alias(name)]))


def describe(s, label):
    s = s.drop_nulls()
    n = s.len()
    if n == 0:
        return f"{label}: n=0"
    qs = [s.quantile(p) for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)]
    return (f"{label}: n={n:,} pos={int((s > 0).sum()):,} zero={int((s == 0).sum()):,} "
            f"neg={int((s < 0).sum()):,} median={s.median()} "
            f"p01/p05/p25/p50/p75/p95/p99={[round(float(x), 1) for x in qs]}")


for label in ["2024", "2026"]:
    print(f"\n==================== {label} ====================")
    rst = pl.scan_parquet(parts(label, "round_state")).select(
        ["match_key", "round", "tick", "event_type"]).collect()
    fe = first_key(rst.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]), "fe")
    bomb = {}
    for et in ["bomb_planted", "bomb_defused", "bomb_exploded"]:
        bomb[et] = first_key(rst.filter(pl.col("event_type") == et).select(["match_key", "round", "tick"]), "b")
    E = first_key(pl.scan_parquet(parts(label, "round_end")).select(["match_key", "round", "tick"]).collect(), "E")
    S = first_key(pl.scan_parquet(parts(label, "round_start")).select(["match_key", "round", "tick"]).collect(), "S")

    print(f"keys: freeze_end={fe.height:,} round_end={E.height:,} round_start={S.height:,}")

    # 1. E - fe
    j = E.join(fe, on=["match_key", "round"], how="inner")
    d_ef = (j["E"] - j["fe"])
    print("[1] E(r) - fe(r) ticks: " + describe(d_ef, label))
    print("    E(r) - fe(r) seconds: " + describe(d_ef / TPS, label))
    print(f"    keys with both fe and E: {j.height:,}; of all E keys {E.height:,}")

    # 2. bomb containment
    for et, b in bomb.items():
        jb = b.join(fe, on=["match_key", "round"], how="inner").join(E, on=["match_key", "round"], how="inner")
        inside = jb.filter((pl.col("b") > pl.col("fe")) & (pl.col("b") <= pl.col("E"))).height
        before_fe = jb.filter(pl.col("b") <= pl.col("fe")).height
        after_E = jb.filter(pl.col("b") > pl.col("E")).height
        print(f"[2] {et}: keys with fe and E {jb.height:,}; inside (fe, E] {inside:,} "
              f"({inside / max(jb.height, 1):.4f}); at or before fe {before_fe:,}; after E {after_E:,}")

    # 3. previous-round containment: E(r-1) < fe(r) < E(r)
    Ep = E.with_columns((pl.col("round") + 1).alias("round")).rename({"E": "E_prev"})
    jp = fe.join(Ep, on=["match_key", "round"], how="inner").join(E, on=["match_key", "round"], how="inner")
    ok = jp.filter((pl.col("E_prev") < pl.col("fe")) & (pl.col("fe") < pl.col("E"))).height
    print(f"[3] E(r-1) < fe(r) < E(r): holds in {ok:,} of {jp.height:,} keys with all three")

    # 4. round_state rows keyed to round r: min tick vs fe, max tick vs E
    rng = rst.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("rs_min"),
                                                   pl.col("tick").max().alias("rs_max"))
    k = rng.join(fe, on=["match_key", "round"], how="inner").join(E, on=["match_key", "round"], how="inner")
    print(f"[4] round_state rows keyed r: keys {k.height:,}; rs_min >= fe: "
          f"{k.filter(pl.col('rs_min') >= pl.col('fe')).height:,}; rs_max <= E: "
          f"{k.filter(pl.col('rs_max') <= pl.col('E')).height:,}; rs_min < E(r-1): "
          f"{k.join(Ep, on=['match_key','round'], how='inner').filter(pl.col('rs_min') < pl.col('E_prev')).height:,}")
    print("    rs_min - fe (s): " + describe((k["rs_min"] - k["fe"]) / TPS, label))
    print("    rs_max - E  (s): " + describe((k["rs_max"] - k["E"]) / TPS, label))

    # 5. S - fe (channel under dispute, for comparison)
    js = S.join(fe, on=["match_key", "round"], how="inner")
    print("[5] S(r) - fe(r) seconds: " + describe((js["S"] - js["fe"]) / TPS, label))
