# Probe: event_type vocabularies for bomb and round channels (2024 tome), and tick/second consistency.
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
def scan(ch): return pl.scan_parquet(files(BASE / f"basic_{ch}{SUF}"))
for ch in ["bomb_state", "bomb_defuse", "bomb_action", "round_state"]:
    df = scan(ch).group_by("event_type").len().collect().sort("event_type")
    print(f"== {ch} event_type counts:", df.rows())
# tick vs second consistency in round_end and bomb_state
re_ = scan("round_end").select(["match_key", "round", "tick", "second"]).collect()
print("round_end tick/second ratio (tick/second) describe:")
print(re_.with_columns((pl.col("tick") / pl.col("second")).alias("r")).select(pl.col("r").describe()))
bs = scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused", "bomb_planted"])).select(["match_key", "round", "tick", "second", "event_type"]).collect()
print("bomb_state tick/second ratio describe:")
print(bs.with_columns((pl.col("tick") / pl.col("second")).alias("r")).select(pl.col("r").describe()))
