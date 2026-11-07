"""verify_D2_9 side check: does the 2024 round_end contamination (class B, see verify_D2_9_chk_endsrc.py)
line up with the open round-13 winner/bomb mismatch? Read-only on the tomes.

ASSUMPTIONS (marked):
  - bomb outcome per (match, round): first round_state event among bomb_exploded (-> T, code 2) or
    bomb_defused (-> CT, code 3). Mismatch = recorded winner_team_code (basic_round_end) differs from that code.
  - class from endsrc: A = basic_round_end tick == round_state round_end; B = basic_round_end tick ==
    round_state cs_pre_restart and no round_state round_end on the key; C = other.
Reports mismatch counts by class for round 13 and for all other rounds, for 2024 and 2026 (control).
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_chk_r13link.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def parts(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
    assert len(fd) == 1, (label, ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def first_key(df, name):
    """First tick per (match_key, round), renamed to `name`."""
    return df.sort("tick").unique(["match_key", "round"], keep="first").rename({"tick": name})


for label in ["2024", "2026"]:
    print(f"\n==================== {label} ====================")
    rst = pl.scan_parquet(parts(label, "round_state")).select(
        ["match_key", "round", "tick", "event_type"]).collect()

    def ev(et, name):
        return first_key(rst.filter(pl.col("event_type") == et).select(["match_key", "round", "tick"]), name)

    re_ch = pl.scan_parquet(parts(label, "round_end")).select(
        ["match_key", "round", "tick", "winner_team_code"]).collect()
    E = first_key(re_ch.select(["match_key", "round", "tick"]), "E")
    W = re_ch.sort("tick").unique(["match_key", "round"], keep="first").select(
        ["match_key", "round", pl.col("winner_team_code").alias("W")])
    RE = ev("round_end", "RE")
    CPR = ev("cs_pre_restart", "CPR")
    bx = ev("bomb_exploded", "bx")
    bd = ev("bomb_defused", "bd")

    base = (E.join(W, on=["match_key", "round"]).join(RE, on=["match_key", "round"], how="left")
            .join(CPR, on=["match_key", "round"], how="left")
            .join(bx, on=["match_key", "round"], how="left").join(bd, on=["match_key", "round"], how="left"))
    base = base.with_columns(
        pl.when((pl.col("E") == pl.col("RE"))).then(pl.lit("A"))
        .when((pl.col("E") == pl.col("CPR")) & pl.col("RE").is_null()).then(pl.lit("B"))
        .otherwise(pl.lit("C")).alias("cls"),
        pl.when(pl.col("bx").is_not_null() & pl.col("bd").is_null()).then(pl.lit(2))
        .when(pl.col("bd").is_not_null() & pl.col("bx").is_null()).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("bomb_code"))
    # rows with a bomb outcome and a recorded winner
    ob = base.filter(pl.col("bomb_code").is_not_null() & pl.col("W").is_not_null())
    ob = ob.with_columns((pl.col("W") != pl.col("bomb_code")).alias("mismatch"),
                         pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("grp"))
    print(f"keys with bomb outcome and winner: {ob.height:,}")
    print(ob.group_by(["grp", "cls"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mismatch"))
          .with_columns((pl.col("mismatch") / pl.col("n")).round(4).alias("rate")).sort(["grp", "cls"]))
