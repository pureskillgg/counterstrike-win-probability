"""D6-4 verifier, step 13: how big is the round-13 inversion population beyond the 434 bomb-verified rows?
Uses the per-round join from step 2 (score side from freeze_end deltas, side-attached, no swap - see step 3).
Splits round 13 (2024, player_count 20, days 12-25) by whether a bomb outcome exists, and reports the share where
round_end disagrees with the score side. Also the same split for days 01-07 as a baseline.
Prints counts only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
j = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet")).filter(pl.col("player_count") == 20)
d = j.filter(pl.col("round") == 13).with_columns(
    pl.when(pl.col("day").is_in([f"{x:02d}" for x in range(12, 26)])).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("win"),
    pl.col("bomb_side").is_not_null().alias("has_bomb"))
for win in ["d01-07", "d12-25"]:
    for hb in [True, False]:
        s = d.filter((pl.col("win") == win) & (pl.col("has_bomb") == hb) & pl.col("score_side").is_not_null())
        dis = s.filter(pl.col("score_side") != pl.col("winner_team_code")).height
        print(f"{win} has_bomb={hb}: round-13 rows={d.filter((pl.col('win')==win)&(pl.col('has_bomb')==hb)).height}, "
              f"score known={s.height}, round_end != score={dis} ({(dis/s.height*100 if s.height else 0):.1f}%)")
