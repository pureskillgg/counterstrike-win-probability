"""D7-6 probe: sample ledger timelines (round_state round_freeze_end t_score/ct_score) vs winner and bomb side.
Shows the score convention before running the full check. Two matches: one pre-onset, one post-onset with r13 mismatch.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
RS = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_state{SUF}")]).filter(
    pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).collect()
print("round_freeze_end rows:", RS.height, "unique (match,round):", RS.select(["match_key", "round"]).unique().height)
R = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_all.parquet"))
R = R.filter(pl.col("player_count") == 20)
pre_key = R.filter((pl.col("mdt") < pl.datetime(2024, 2, 6)) & (pl.col("round").is_between(10, 16)) & (pl.col("bomb_side").is_not_null())).sort("mdt")["match_key"][0]
post_key = R.filter((pl.col("mdt") > pl.datetime(2024, 2, 9)) & (pl.col("round") == 13) & pl.col("mm")).sort("mdt")["match_key"][0]
for label, k in [("PRE", pre_key), ("POST", post_key)]:
    print(f"\n== {label}: {k[:60]} ==")
    rr = R.filter(pl.col("match_key") == k).select(["round", "winner_team_code", "bomb_side", "planter_side", "mm", "lag" if False else "out_tick", "tick"]).sort("round")
    rs = RS.filter(pl.col("match_key") == k).sort("round")
    j = rs.join(rr, on="round", how="full", coalesce=True).sort("round")
    print(j.select(["round", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "winner_team_code", "bomb_side", "planter_side", "mm", "tick", "out_tick"]))
