"""D6-4 verifier: event-type and sample probe, 2024 tome (lazy, projected)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def scan(ch):
    return pl.scan_parquet(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*"))
rs = scan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase", "is_warmup"]).collect()
print("round_state event types:")
print(rs.group_by("event_type").len().sort("len", descending=True))
bs = scan("bomb_state").select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
print("bomb_state event types:")
print(bs.group_by("event_type").len().sort("len", descending=True))
# sample match with bomb rounds
mk = (bs.filter(pl.col("event_type") == "bomb_exploded").select("match_key").unique().head(1)).item()
print("sample match_key:", mk)
print(rs.filter(pl.col("match_key") == mk).sort(["round", "tick"]).head(80))
print(bs.filter(pl.col("match_key") == mk).sort(["round", "tick"]).head(40))
re_ = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"]).filter(pl.col("match_key") == mk).sort("round").collect()
print(re_)
