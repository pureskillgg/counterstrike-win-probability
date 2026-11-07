"""verify_D2_3_08: 2026 control. Same bomb-only truth as verify_D2_3_01 (A1-A4), applied to the 2026-10-01 to 08 tome,
which is the bake-off's data (scripts\\model\\bakeoff.py). Also the same build check and the same round-bin flip table.

Reads the 2026 tome only. Column projection; one channel at a time. No files written.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D2_3_08_2026_control.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
MK = "match_key"
TOL = 40  # ASSUMPTION A1


def files(ch):
    fd = sorted(glob.glob(str(BASE / f"basic_{ch}.*")))
    assert len(fd) == 1, (ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


hdr = pl.read_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*"))),
                      columns=["key", "build_num", "map_name"]).rename({"key": MK})
re_raw = pl.scan_parquet(files("round_end")).select([MK, "round", "tick", "winner_team_code", "player_count"]).collect()
re_ = re_raw.sort("tick").unique([MK, "round"], keep="first").rename({"tick": "t_end"})
rs = (pl.scan_parquet(files("round_state")).select([MK, "round", "tick", "event_type"])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
fe = rs.sort("tick").unique([MK, "round"], keep="first").select([MK, "round", pl.col("tick").alias("t_fe")])
bs = pl.scan_parquet(files("bomb_state")).select([MK, "round", "tick", "event_type"]).collect()
bs = (bs.join(fe, on=[MK, "round"], how="left").join(re_.select([MK, "round", "t_end"]), on=[MK, "round"], how="left")
      .with_columns(((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + TOL)).alias("in_win")))
ob = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]) & pl.col("in_win"))
      .sort("tick").group_by([MK, "round"]).agg(pl.col("event_type").first().alias("bo_first"))
      .with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
                    .cast(pl.Int64).alias("B")))
k = (re_.select([MK, "round", "winner_team_code"]).join(ob.select([MK, "round", "B"]), on=[MK, "round"], how="inner")
     .join(hdr, on=MK, how="left").with_columns((pl.col("winner_team_code") != pl.col("B")).alias("flip")))
print("2026 rounds with bomb-only truth:", k.height, " flips:", k.filter("flip").height)
print("by round (flips, n) for rounds 1-30:")
print(k.group_by("round").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"))
      .filter(pl.col("round") <= 30).sort("round").rows())
print("by build_num:")
print(k.group_by("build_num").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped")).sort("build_num"))
print("end of 2026 control (no files written)")
