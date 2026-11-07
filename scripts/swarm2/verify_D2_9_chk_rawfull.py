"""verify_D2_9 full tome-vs-raw comparison for 2024 (read-only on E:, nothing written).

For every 2024 (match, round) in the tome's basic_round_end and basic_round_start channels, read the matching
raw match folder on E: (csds_tail_recovered_20261011\\csds\\2024\\02\\<DD>\\<uuid>\\{round_end,round_start,round_state})
and compare first-tick values per round. Reports:
  - how many tome match keys have a raw folder with the needed files;
  - tome basic_round_end == raw round_end tick (per round);
  - tome basic_round_start == raw round_start tick (per round);
  - among rounds where tome round_end != raw round_state round_end (or no raw round_end state event), whether
    the tome round_end equals raw cs_pre_restart (the reset-time event).
Output is summarized; per-match detail is not written to disk.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_chk_rawfull.py
"""
import glob
import sys
import time
from collections import Counter
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
RAW24 = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")


def parts_tome(ch):
    fd = sorted(glob.glob(str(TOME24 / f"basic_{ch}.*")))
    assert len(fd) == 1, (ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def first_key(df, name):
    return (df.sort("tick").unique(["match_key", "round"], keep="first")
            .select(["match_key", "round", pl.col("tick").alias(name)]))


t0 = time.time()
tE = first_key(pl.scan_parquet(parts_tome("round_end")).select(["match_key", "round", "tick"]).collect(), "tE")
tS = first_key(pl.scan_parquet(parts_tome("round_start")).select(["match_key", "round", "tick"]).collect(), "tS")
match_keys = sorted(tE["match_key"].unique().to_list())
print(f"tome 2024: basic_round_end keys={tE.height:,} matches={len(match_keys):,}; "
      f"basic_round_start keys={tS.height:,}")

rows = []          # per-round records
missing = Counter()
for i, mk in enumerate(match_keys):
    p = mk.split("/")  # csds / 2024 / 02 / DD / uuid / csds
    md = RAW24 / p[3] / p[4]
    fe_ = md / "round_end"
    fs_ = md / "round_start"
    fst = md / "round_state"
    if not (fe_.is_file() and fs_.is_file() and fst.is_file()):
        missing["match folder or required channel missing"] += 1
        continue
    try:
        re_raw = pl.read_parquet(str(fe_), columns=["round", "tick"])
        rs_raw = pl.read_parquet(str(fs_), columns=["round", "tick"])
        st_raw = pl.read_parquet(str(fst), columns=["round", "tick", "event_type"]).filter(
            pl.col("event_type").is_in(["round_end", "cs_pre_restart", "round_freeze_end", "round_start"]))
    except Exception as ex:  # noqa: BLE001 - count and continue
        missing[f"read error: {type(ex).__name__}"] += 1
        continue
    re_first = re_raw.sort("tick").unique("round", keep="first").rename({"tick": "rE"})
    rs_first = rs_raw.sort("tick").unique("round", keep="first").rename({"tick": "rS"})
    st_end = (st_raw.filter(pl.col("event_type") == "round_end").sort("tick").unique("round", keep="first")
              .select(["round", pl.col("tick").alias("sRE")]))
    st_cpr = (st_raw.filter(pl.col("event_type") == "cs_pre_restart").sort("tick").unique("round", keep="first")
              .select(["round", pl.col("tick").alias("sCPR")]))
    st_fe = (st_raw.filter(pl.col("event_type") == "round_freeze_end").sort("tick").unique("round", keep="first")
             .select(["round", pl.col("tick").alias("sFE")]))
    j = (re_first.join(rs_first, on="round", how="full", coalesce=True)
         .join(st_end, on="round", how="left").join(st_cpr, on="round", how="left").join(st_fe, on="round", how="left")
         .with_columns(pl.lit(mk).alias("match_key")))
    rows.append(j)
    if (i + 1) % 500 == 0:
        print(f"  processed {i + 1:,} matches ({time.time() - t0:.0f}s)")

print(f"raw matches read: {len(rows):,} of {len(match_keys):,}; missing/unreadable: {dict(missing)}")
raw = pl.concat(rows, how="vertical_relaxed")
raw = raw.rename({"round": "round"})
cmp_ = tE.join(raw, left_on=["match_key", "round"], right_on=["match_key", "round"], how="inner")
print(f"tome basic_round_end keys with raw round_end: {cmp_.height:,}")
print(f"  tome E == raw round_end tick: {cmp_.filter(pl.col('tE') == pl.col('rE')).height:,}")
print(f"  tome E != raw round_end tick: {cmp_.filter(pl.col('tE') != pl.col('rE')).height:,}")

cmpS = tS.join(raw, on=["match_key", "round"], how="inner")
print(f"tome basic_round_start keys with raw round_start: {cmpS.height:,}")
print(f"  tome S == raw round_start tick: {cmpS.filter(pl.col('tS') == pl.col('rS')).height:,}")

# Classify tome E by raw round_state presence
cls = cmp_.with_columns(
    pl.when(pl.col("sRE").is_not_null() & (pl.col("tE") == pl.col("sRE"))).then(pl.lit("A tome E == raw state round_end"))
    .when(pl.col("sCPR").is_not_null() & (pl.col("tE") == pl.col("sCPR"))).then(pl.lit("B tome E == raw state cs_pre_restart (no state round_end)"))
    .when(pl.col("tE") == pl.col("rE")).then(pl.lit("C tome E == raw round_end channel only"))
    .otherwise(pl.lit("D other")).alias("cls"))
print("\nclassification of tome E (keys with raw data):")
print(cls.group_by("cls").len().sort("cls"))

# Among class B, is the raw round_end channel itself equal to cs_pre_restart?
b = cls.filter(pl.col("cls").str.starts_with("B"))
if b.height:
    print(f"class B keys: {b.height:,}; raw round_end channel == raw cs_pre_restart: "
          f"{b.filter(pl.col('rE') == pl.col('sCPR')).height:,}; raw round_state round_end present: "
          f"{b.filter(pl.col('sRE').is_not_null()).height:,}")
    d = ((b["tE"] - b["sFE"]).cast(pl.Float64) / 64.0).drop_nulls()
    print(f"class B: E - freeze_end seconds median={d.median():.2f} p05={d.quantile(0.05):.2f} p95={d.quantile(0.95):.2f}")
a = cls.filter(pl.col("cls").str.starts_with("A"))
if a.height:
    d = ((a["tE"] - a["sFE"]).cast(pl.Float64) / 64.0).drop_nulls()
    print(f"class A: E - freeze_end seconds median={d.median():.2f} p05={d.quantile(0.05):.2f} p95={d.quantile(0.95):.2f}")
print(f"\nelapsed {time.time() - t0:.0f}s")
