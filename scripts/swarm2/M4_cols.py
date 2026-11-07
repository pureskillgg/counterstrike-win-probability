"""M4: print column names for the channels used by the map and site analysis (footers only)."""
import glob
import os

import pyarrow.parquet as pq

BASE = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
SUFFIX = ".2026-09-18,2026-10-08"
for ch in ["bomb_state", "bomb_action", "bomb_defuse", "round_end", "round_start", "round_state",
           "player_death", "player_info", "player_spawn", "team_change", "player_disconnect",
           "bot_takeover"]:
    folder = os.path.join(BASE, f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    print(ch, pq.read_schema(files[0]).names)
print("header", pq.read_schema(sorted(glob.glob(os.path.join(BASE, "header", "dataframe_*")))[0]).names)
