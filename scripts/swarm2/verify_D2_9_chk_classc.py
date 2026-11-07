"""verify_D2_9 follow-up: characterize the 2024 round_end classes inside matches that contain class-B rounds.
Read-only on the 2024 tome. Same class definitions as verify_D2_9_chk_endsrc.py:
  A: basic_round_end tick == round_state round_end (same key)
  B: basic_round_end tick == round_state cs_pre_restart and no round_state round_end on the key
  C: neither
Reports: for matches with >=1 class-B round, the round numbers of class-C and class-A rows relative to the
match's last round, and the match-date range of matches with class B versus without.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_chk_classc.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
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
E = first_key(pl.scan_parquet(parts("round_end")).select(["match_key", "round", "tick"]).collect(), "E")
RE = first_key(rst.filter(pl.col("event_type") == "round_end").select(["match_key", "round", "tick"]), "RE")
CPR = first_key(rst.filter(pl.col("event_type") == "cs_pre_restart").select(["match_key", "round", "tick"]), "CPR")
base = (E.join(RE, on=["match_key", "round"], how="left").join(CPR, on=["match_key", "round"], how="left")
        .with_columns(
            pl.when(pl.col("E") == pl.col("RE")).then(pl.lit("A"))
            .when((pl.col("E") == pl.col("CPR")) & pl.col("RE").is_null()).then(pl.lit("B"))
            .otherwise(pl.lit("C")).alias("cls"),
            pl.col("match_key").str.split("/").list.get(3).alias("day")))
mx = base.group_by("match_key").agg(pl.col("round").max().alias("last_round"), pl.len().alias("n_rounds"),
                                    (pl.col("cls") == "B").sum().alias("n_B"))
b = base.join(mx, on="match_key")
has_b = mx.filter(pl.col("n_B") > 0)
print(f"matches with >=1 class-B round: {has_b.height:,}; matches without: {mx.height - has_b.height:,}")
print("day range of matches with class B:", sorted(b.filter(pl.col('n_B') > 0)["day"].unique().to_list()))
print("day range of matches without class B:", sorted(b.filter(pl.col('n_B') == 0)["day"].unique().to_list()))

bb = b.filter(pl.col("n_B") > 0)
print("\nin class-B matches, rows by class and whether round == last round of the match:")
print(bb.with_columns((pl.col("round") == pl.col("last_round")).alias("is_last")).group_by(["cls", "is_last"]).len().sort(["cls", "is_last"]))
print("\nclass-C rows in class-B matches: round minus last round")
print(bb.filter(pl.col("cls") == "C").with_columns((pl.col("round") - pl.col("last_round")).alias("off"))
      .group_by("off").len().sort("off"))
print("\nclass-A rows in class-B matches: count per match (distribution)")
nA = (bb.filter(pl.col("cls") == "A").group_by("match_key").agg(pl.len().alias("n_A")))
print(f"class-B matches with >=1 class-A row: {nA.height:,}; n_A per match quantiles (min, median, max): "
      f"{nA['n_A'].min()}, {nA['n_A'].median()}, {nA['n_A'].max()}")
print("class-A rows in class-B matches: total", int(nA['n_A'].sum()) if nA.height else 0)
print("\nmatches in class-B set: rounds per match (min, median, max):",
      has_b["n_rounds"].min(), has_b["n_rounds"].median(), has_b["n_rounds"].max())
