"""D9 schema probe: columns of 2024 header and event channels (read-only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"

def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))

targets = {
    "header": BASE / "header",
    "round_end": BASE / f"basic_round_end{SUF}",
    "round_start": BASE / f"basic_round_start{SUF}",
    "round_state": BASE / f"basic_round_state{SUF}",
    "bomb_state": BASE / f"basic_bomb_state{SUF}",
    "bomb_action": BASE / f"basic_bomb_action{SUF}",
    "bomb_defuse": BASE / f"basic_bomb_defuse{SUF}",
    "bot_takeover": BASE / f"basic_bot_takeover{SUF}",
    "player_disconnect": BASE / f"basic_player_disconnect{SUF}",
    "player_info": BASE / f"basic_player_info{SUF}",
    "player_personal": BASE / f"basic_player_personal{SUF}",
}
for name, fd in targets.items():
    fs = files_of(fd)
    if not fs:
        print(name, "NO dataframe files at", fd)
        continue
    sch = pl.read_parquet_schema(fs[0])
    print(f"== {name}: {len(fs)} file(s); first={Path(fs[0]).name}")
    for k, v in sch.items():
        print(f"   {k}: {v}")
    lf = pl.scan_parquet(fs)
    print("   rows(lazy count):", lf.select(pl.len()).collect().item())
