"""D2-7 verifier: schema and event-type probe of the 2024 tome (column-projected, read-only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files_of(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


for ch in ("round_end", "round_state", "bomb_state", "player_info", "player_death"):
    fs = files_of(ch)
    sch = pl.read_parquet_schema(fs[0])
    n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in fs)
    print(f"{ch}: files={len(fs)} rows={n}")
    print("   schema:", dict(sch))

rs = pl.concat([pl.read_parquet(f, columns=["event_type"]) for f in files_of("round_state")])
print("round_state event_type counts:", rs.group_by("event_type").len().sort("event_type").rows())
bs = pl.concat([pl.read_parquet(f, columns=["event_type"]) for f in files_of("bomb_state")])
print("bomb_state event_type counts:", bs.group_by("event_type").len().sort("event_type").rows())
re_ = pl.concat([pl.read_parquet(f, columns=["winner_team_code", "win_reason_code", "win_reason_message", "legacy_code"]) for f in files_of("round_end")])
print("round_end winner_team_code counts:", re_.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("round_end win_reason_code x message (all rounds):")
print(re_.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))
