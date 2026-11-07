"""D6-5 verifier, part 7: raw round_state / round_end / player_death events for round 13 of one match (read-only)."""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\19\40f9484a-d796-42b2-b32e-dead7d12bc39")
rs = pl.read_parquet(BASE / "round_state")
print("raw round_state columns:", rs.columns)
ev = rs.filter(pl.col("round").is_in([12, 13, 14]) & pl.col("event_type").is_in(
    ["round_freeze_end", "round_end", "cs_win_panel_round", "bomb_planted", "bomb_defused", "bomb_exploded", "round_start"]))
cols = [c for c in ["round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"] if c in ev.columns]
print(ev.select(cols).sort(["round", "tick"]))
try:
    cwp = pl.read_parquet(BASE / "round_state")  # same file; winner side fields if present
except Exception as e:
    print("err", e)
for ch in ["player_death", "bomb_state", "player_info"]:
    try:
        df = pl.read_parquet(BASE / ch)
        print(f"raw {ch}: rows {df.height}, round-13 rows {df.filter(pl.col('round') == 13).height}")
    except Exception as e:
        print(f"raw {ch}: read failed {type(e).__name__}")
