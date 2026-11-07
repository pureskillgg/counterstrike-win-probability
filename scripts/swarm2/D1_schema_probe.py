import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
chans = ["player_info", "player_spawn", "round_state", "round_start", "round_end", "bomb_state",
         "player_name", "player_personal", "round_mvp", "player_death", "player_footstep", "bomb_action", "bomb_defuse", "other_death"]
for ch in chans:
    fd = BASE / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    if not files:
        print(ch, "NO FILES", fd.exists())
        continue
    sch = pl.read_parquet_schema(files[0])
    print(f"{ch}: nfiles={len(files)} cols={list(sch.items())}")
