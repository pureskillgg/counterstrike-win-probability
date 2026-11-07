"""verify_D2_9 match-level clustering of 2024 round_end classes (A/B/C from verify_D2_9_chk_endsrc.py),
and day-level share of class B versus the round-13 mismatch rate from verify_D2_9_chk_r13link.py.
Read-only on the 2024 tome. ASSUMPTIONS as in verify_D2_9_chk_endsrc.py and verify_D2_9_chk_r13link.py.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_chk_clusters.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def parts(ch):
    fd = sorted(glob.glob(str(TOME24 / f"basic_{ch}.*")))
    assert len(fd) == 1, (ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def first_key(df, name):
    return df.sort("tick").unique(["match_key", "round"], keep="first").rename({"tick": name})


rst = pl.scan_parquet(parts("round_state")).select(["match_key", "round", "tick", "event_type"]).collect()
re_ch = pl.scan_parquet(parts("round_end")).select(["match_key", "round", "tick", "winner_team_code"]).collect()
E = first_key(re_ch.select(["match_key", "round", "tick"]), "E")
W = re_ch.sort("tick").unique(["match_key", "round"], keep="first").select(
    ["match_key", "round", pl.col("winner_team_code").alias("W")])
RE = first_key(rst.filter(pl.col("event_type") == "round_end").select(["match_key", "round", "tick"]), "RE")
CPR = first_key(rst.filter(pl.col("event_type") == "cs_pre_restart").select(["match_key", "round", "tick"]), "CPR")
bx = first_key(rst.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round", "tick"]), "bx")
bd = first_key(rst.filter(pl.col("event_type") == "bomb_defused").select(["match_key", "round", "tick"]), "bd")

base = (E.join(W, on=["match_key", "round"]).join(RE, on=["match_key", "round"], how="left")
        .join(CPR, on=["match_key", "round"], how="left")
        .join(bx, on=["match_key", "round"], how="left").join(bd, on=["match_key", "round"], how="left"))
base = base.with_columns(
    pl.when(pl.col("E") == pl.col("RE")).then(pl.lit("A"))
    .when((pl.col("E") == pl.col("CPR")) & pl.col("RE").is_null()).then(pl.lit("B"))
    .otherwise(pl.lit("C")).alias("cls"),
    pl.when(pl.col("bx").is_not_null() & pl.col("bd").is_null()).then(pl.lit(2))
    .when(pl.col("bd").is_not_null() & pl.col("bx").is_null()).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("bomb_code"))
base = base.with_columns(
    pl.col("match_key").str.split("/").list.get(3).alias("day"),
    (pl.col("bomb_code").is_not_null() & pl.col("W").is_not_null() & (pl.col("W") != pl.col("bomb_code"))).alias("mismatch"),
    (pl.col("bomb_code").is_not_null() & pl.col("W").is_not_null()).alias("has_outcome"))

# 1. match-level clustering of class B
per_match = base.group_by("match_key").agg(
    pl.len().alias("n_rounds"),
    (pl.col("cls") == "B").sum().alias("n_B"),
    (pl.col("cls") == "A").sum().alias("n_A"))
per_match = per_match.with_columns((pl.col("n_B") / pl.col("n_rounds")).alias("fracB"))
print(f"2024 matches with rounds in basic_round_end: {per_match.height:,}")
print("matches by class-B share (0, 0<x<1, 1):")
print(per_match.with_columns(
    pl.when(pl.col("n_B") == 0).then(pl.lit("0 none"))
    .when(pl.col("n_B") == pl.col("n_rounds")).then(pl.lit("1 all rounds B"))
    .otherwise(pl.lit("2 mixed")).alias("bucket")).group_by("bucket").len().sort("bucket"))
print("class-B share histogram by decile of fracB:")
print(per_match.with_columns((pl.col("fracB") * 10).floor().clip(0, 9).cast(pl.Int64).alias("dec"))
      .group_by("dec").len().sort("dec"))

# 2. day-level: class-B share and round-13 mismatch rate
day = base.group_by("day").agg(
    pl.len().alias("rounds"),
    (pl.col("cls") == "B").sum().alias("n_B"),
    pl.col("has_outcome").sum().alias("outcome_rows"),
    (pl.col("mismatch") & (pl.col("round") == 13)).sum().alias("r13_mismatch"),
    ((pl.col("round") == 13) & pl.col("has_outcome")).sum().alias("r13_outcome_rows"),
    (pl.col("mismatch") & (pl.col("round") != 13)).sum().alias("other_mismatch"),
).with_columns(
    (pl.col("n_B") / pl.col("rounds")).round(3).alias("shareB"),
    (pl.col("r13_mismatch") / pl.col("r13_outcome_rows")).round(3).alias("r13_mismatch_rate")).sort("day")
print("\nper day (2024): rounds, class-B share, round-13 outcome rows, round-13 mismatch rate, other mismatches")
print(day)

# 3. class-B rounds among round 13 with bomb outcome, and whether round_state round_end is absent for the match
r13 = base.filter((pl.col("round") == 13) & pl.col("has_outcome"))
print("\nround 13 with outcome by class:")
print(r13.group_by("cls").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mismatch")).sort("cls"))
