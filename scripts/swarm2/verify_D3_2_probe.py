"""D3-2 verifier probe: schemas, event vocabularies, tick rate. Reads only the G: 2024 tome."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


for ch in ["round_end", "bomb_state", "round_state"]:
    fs = files(ch)
    print(f"== {ch}: {len(fs)} files")
    print(pl.read_parquet_schema(fs[0]))

bs = pl.concat([pl.read_parquet(f, columns=["event_type", "player_team_code"]) for f in files("bomb_state")])
print("\nbomb_state event_type counts:")
print(bs.group_by("event_type").len().sort("len", descending=True))
rs = pl.concat([pl.read_parquet(f, columns=["event_type"]) for f in files("round_state")])
print("\nround_state event_type counts:")
print(rs.group_by("event_type").len().sort("len", descending=True))
re_ = pl.concat([pl.read_parquet(f, columns=["winner_team_code", "win_reason_code", "win_reason_message", "legacy_code", "player_count"]) for f in files("round_end")])
print("\nround_end player_count counts:")
print(re_.group_by("player_count").len().sort("len", descending=True))
print("\nround_end win_reason_code x message:")
print(re_.group_by(["win_reason_code", "win_reason_message"]).len().sort("win_reason_code"))
print("\nround_end winner_team_code counts:")
print(re_.group_by("winner_team_code").len())
