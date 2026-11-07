"""D6-4 verifier, step 19: win_reason_code by bomb outcome for round 13 (2024, player_count 20, days 12-25),
plus the tome round_end reason for the affected rows. Counts only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
rend = (pl.scan_parquet(str([p for p in base.iterdir() if p.name.startswith("basic_round_end.")][0] / "dataframe_*"))
        .select(["match_key", "round", "winner_team_code", "win_reason_code"]).collect())
j = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet")).filter((pl.col("player_count") == 20) & (pl.col("round") == 13)
                                                                     & pl.col("day").is_in([f"{x:02d}" for x in range(12, 26)]))
j = j.join(rend.select(["match_key", "round", "win_reason_code"]), on=["match_key", "round"], how="left")
b = j.filter(pl.col("bomb_side").is_not_null()).with_columns(
    pl.when(pl.col("bomb_side") == 2).then(pl.lit("exploded")).otherwise(pl.lit("defused")).alias("bomb"))
print("round 13 d12-25 bomb-known rows x reason code (round_end):")
print(b.group_by(["bomb", "win_reason_code"]).len().sort(["bomb", "win_reason_code"]).rows())
print("round 13 d12-25 bomb-known rows x round_end winner:")
print(b.group_by(["bomb", "winner_team_code"]).len().sort(["bomb", "winner_team_code"]).rows())
