"""D1-10 verifier: round_end win_reason of the 2024 round-13 bomb-vs-winner conflicts.

Takes 2024 tome round-13 rows whose bomb outcome (bomb_exploded only) is in the live window and whose
round_end winner is CT (3), and tabulates the round_end win_reason_code. Also prints the same
tabulation for the agreeing rounds (bomb_exploded and winner T) as a control.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(10)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")


def fl(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}*" / "dataframe_*")))


re_ = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "win_reason_code"]).collect() for f in fl("round_end")])
re_ = re_.unique(["match_key", "round"])
bs = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "event_type"]).collect() for f in fl("bomb_state")])
ex = bs.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("exploded"))
df = fl("bomb_state")
de = bs.filter(pl.col("event_type") == "bomb_defused").select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("defused"))
j = re_.join(ex, on=["match_key", "round"], how="left").join(de, on=["match_key", "round"], how="left").with_columns(
    pl.col("exploded").fill_null(False), pl.col("defused").fill_null(False))
r13 = j.filter(pl.col("round") == 13)
conf = r13.filter(pl.col("exploded") & ~pl.col("defused") & (pl.col("winner_team_code") == 3))
ctrl = r13.filter(pl.col("exploded") & ~pl.col("defused") & (pl.col("winner_team_code") == 2))
print("round 13, 2024: exploded-only rounds:", r13.filter(pl.col("exploded") & ~pl.col("defused")).height)
print("  conflicts (exploded, winner CT):", conf.height)
print("  reason codes on conflicts:", sorted(conf.group_by("win_reason_code").len().rows()))
print("  control (exploded, winner T):", ctrl.height)
print("  reason codes on control:", sorted(ctrl.group_by("win_reason_code").len().rows()))
print("  rounds with both exploded and defused flags (round 13):", r13.filter(pl.col("exploded") & pl.col("defused")).height)
dconf = r13.filter(pl.col("defused") & ~pl.col("exploded") & (pl.col("winner_team_code") == 2))
dctrl = r13.filter(pl.col("defused") & ~pl.col("exploded") & (pl.col("winner_team_code") == 3))
print("  defused-only round 13 with winner T (conflict):", dconf.height, " reason codes:", sorted(dconf.group_by("win_reason_code").len().rows()))
print("  defused-only round 13 with winner CT (control):", dctrl.height, " reason codes:", sorted(dctrl.group_by("win_reason_code").len().rows()))
print("  total round-13 bomb-outcome conflicts (exploded->CT plus defused->T):", conf.height + dconf.height)
r9 = j.filter(pl.col("round") == 9)
c9 = r9.filter(pl.col("exploded") & ~pl.col("defused") & (pl.col("winner_team_code") == 3))
print("round 9, 2024: exploded-only with winner CT:", c9.height, " reason codes:", sorted(c9.group_by("win_reason_code").len().rows()))
print("DONE")
