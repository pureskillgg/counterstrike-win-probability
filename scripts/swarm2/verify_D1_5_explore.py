"""verify D1-5 exploration (swarm2 verifier). Reads only; prints to stdout."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def rd(ch, cols=None):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


RE = rd("round_end")
print("round_end rows", RE.height, "unique (match,round)", RE.select(["match_key", "round"]).unique().height)
print(RE.group_by(["winner_team_code", "win_reason_code", "win_reason_message"]).len().sort("len", descending=True))
print(RE.group_by("legacy_code").len().sort("len", descending=True).head(10))
print(RE.group_by("player_count").len().sort("len", descending=True).head(10))

RS = rd("round_state")
print("\nround_state rows", RS.height)
print(RS.group_by("event_type").len().sort("len", descending=True))
print(RS.group_by("phase").len().sort("len", descending=True).head(12))
print(RS.group_by("is_warmup").len())
d = RS.with_columns((pl.col("t_score") - pl.col("t_score_raw")).alias("dt"), (pl.col("ct_score") - pl.col("ct_score_raw")).alias("dct"))
print("t_score - t_score_raw:", d.group_by("dt").len().sort("dt").to_dicts()[:10])
print("ct_score - ct_score_raw:", d.group_by("dct").len().sort("dct").to_dicts()[:10])
print(RS.head(12))

BS = rd("bomb_state")
print("\nbomb_state rows", BS.height)
print(BS.group_by("event_type").len().sort("len", descending=True))
print(BS.group_by(["event_type", "player_team_code"]).len().sort(["event_type", "player_team_code"]))

RST = rd("round_start") if glob.glob(str(BASE / f"basic_round_start{SUF}" / "dataframe_*")) else None
if RST is not None:
    print("\nround_start rows", RST.height, RST.columns)
    print(RST.head(3))
