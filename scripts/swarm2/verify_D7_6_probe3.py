"""verify D7-6 probe3: schemas of player-level tables carrying team codes; win_reason crosstab; event coverage by date."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
for tbl in ["basic_player_spawn", "basic_player_info", "basic_player_death", "basic_round_start", "basic_round_state"]:
    fs = files(BASE / f"{tbl}{SUF}")
    if not fs:
        print("no files", tbl); continue
    lf = pl.concat([pl.scan_parquet(f) for f in fs])
    names = lf.collect_schema().names()
    print(f"== {tbl} rows={lf.select(pl.len()).collect().item()} cols={names}")
RE = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_end{SUF}")]).collect()
print("round_end cols:", RE.columns, "rows", RE.height)
print("win_reason_message counts:", RE.group_by(["win_reason_code", "win_reason_message"]).len().sort(["win_reason_code"]).rows())
print("winner_team_code dist:", RE.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("player_count dist:", RE.group_by("player_count").len().sort("player_count").rows())
