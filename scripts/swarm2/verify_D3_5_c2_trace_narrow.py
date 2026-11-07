"""D3-5 verifier, step C2: narrow per-round trace (freeze t/ct, raw fields, round_end reason, bomb events)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_width_chars(200)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


RE = rd("round_end", ["match_key", "round", "winner_team_code", "tick", "player_count", "win_reason_code"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
fe = RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
fe1 = fe.group_by(["match_key", "round"]).agg(pl.col("tick").first().alias("fe_tick"), pl.col("t_score").first().alias("t"),
                                             pl.col("ct_score").first().alias("ct"),
                                             pl.col("t_score_raw").first().alias("tr"),
                                             pl.col("ct_score_raw").first().alias("cr"))
mats = ["csds/2024/02/12/m_1af6d0dd4935dea0c0e32ccc/csds", "csds/2024/02/12/m_577b8c6808813a48035ddff4/csds"]
for m in mats:
    sub = (RE.filter((pl.col("match_key") == m) & (pl.col("player_count") == 20))
           .join(fe1.filter(pl.col("match_key") == m), on=["match_key", "round"], how="left").sort("round"))
    print("\n", m)
    print(sub.select(["round", "winner_team_code", "win_reason_code", "tick", "fe_tick", "t", "ct", "tr", "cr"]).head(16))
    print(RS.filter((pl.col("match_key") == m) & (pl.col("event_type") == "round_end")).head(4)
          .select(["round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]))
