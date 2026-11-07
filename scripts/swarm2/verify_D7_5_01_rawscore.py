"""verify D7-5 probe: t_score vs t_score_raw in freeze_end rows; duplicates; build boundary. Prints only."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
RS = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_state{SUF}")]).filter(pl.col("event_type") == "round_freeze_end").collect()
print("freeze_end rows:", RS.height)
print("t_score == t_score_raw:", RS.select((pl.col("t_score") == pl.col("t_score_raw")).sum()).item(), "of", RS.height)
print("ct_score == ct_score_raw:", RS.select((pl.col("ct_score") == pl.col("ct_score_raw")).sum()).item(), "of", RS.height)
d = RS.filter((pl.col("t_score") != pl.col("t_score_raw")) | (pl.col("ct_score") != pl.col("ct_score_raw")))
print("rows where raw != corrected:", d.height)
print(d.select(["match_key", "round", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).head(12))
# duplicates in freeze_end
dup = RS.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("duplicate (match,round) in freeze_end:", dup.height)
if dup.height:
    print(RS.join(dup.select(["match_key", "round"]), on=["match_key", "round"]).select(["match_key", "round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "second"]))
RE = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_end{SUF}")]).collect()
dupe = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("duplicate (match,round) in round_end:", dupe.height)
if dupe.height:
    print(RE.join(dupe.select(["match_key", "round"]), on=["match_key", "round"]).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count", "second"]))
# how the two bomb sources line up with round_state bomb rows
BS = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_bomb_state{SUF}")]).collect()
print("bomb_state rows per (match,round,event_type) distribution:")
print(BS.group_by(["match_key", "round", "event_type"]).len().group_by(["event_type", "len"]).len().sort(["event_type", "len"]).rows())
