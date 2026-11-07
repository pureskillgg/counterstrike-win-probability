import sys, glob
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
for ch in ["player_info","player_spawn","round_start","round_state","player_death","bomb_state","round_end","player_personal","round_mvp"]:
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    sch = pl.read_parquet_schema(fs[0])
    n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in fs)
    print(ch, "files", len(fs), "rows", n)
    print("  ", dict(sch))
