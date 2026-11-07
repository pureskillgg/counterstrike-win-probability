"""Verifier probe: schemas, reason vocabulary, map availability (2024 tome). Read-only."""
import glob
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"

def files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))

for ch in ["round_end", "round_state", "round_start", "bomb_state", "bomb_defuse", "bomb_action", "player_info"]:
    fs = files(ch)
    sch = pl.read_parquet_schema(fs[0]) if fs else {}
    print(ch, "files=", len(fs), "cols=", list(sch.keys()))

re_ = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"]) for f in files("round_end")], how="vertical_relaxed")
print("round_end rows", re_.height, "unique (match,round)", re_.select(["match_key", "round"]).unique().height)
print(re_.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))
print("player_count dist", re_.group_by("player_count").len().sort("len", descending=True).head(8))

bs = pl.concat([pl.read_parquet(f, columns=["event_type"]) for f in files("bomb_state")], how="vertical_relaxed")
print(bs.group_by("event_type").len().sort("len", descending=True))

hdr_dir = BASE / "header"
print("header tome files:", sorted(glob.glob(str(hdr_dir / "*")))[:10])
hf = sorted(glob.glob(str(hdr_dir / "dataframe_*")))
if hf:
    hs = pl.read_parquet_schema(hf[0])
    print("header schema", hs)
    hd = pl.read_parquet(hf[0]).head(3)
    print(hd)
