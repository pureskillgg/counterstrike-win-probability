"""verify D7-6 probe: schema and event types of 2024 round_state, round_end, bomb tables (read-only)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
for tbl in ["basic_round_state", "basic_round_end", "basic_bomb_state", "basic_bomb_action", "basic_bomb_defuse", "basic_team_change"]:
    fs = files(BASE / f"{tbl}{SUF}")
    print(f"== {tbl}: files={len(fs)}")
    if not fs:
        continue
    lf = pl.concat([pl.scan_parquet(f) for f in fs])
    print("schema:", {k: str(v) for k, v in lf.collect_schema().items()})
    try:
        if "event_type" in lf.collect_schema().names():
            print("event_type counts:", lf.group_by("event_type").len().sort("event_type").collect().rows())
    except Exception as e:
        print("ERR", e)
    print("rows:", lf.select(pl.len()).collect().item())
