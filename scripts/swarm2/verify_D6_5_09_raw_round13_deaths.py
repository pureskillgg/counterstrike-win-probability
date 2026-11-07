"""D6-5 verifier, part 8: raw round-13 deaths and roster for the same match (read-only)."""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1)
BASE = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\19\m_0fba63102127c34fe07b4f79")
pdth = pl.read_parquet(BASE / "player_death").filter(pl.col("round") == 13)
print("round-13 deaths (tick, victim, victim_team, attacker, attacker_team):")
print(pdth.select(["tick", "player_id", "player_team_code", "attacker_id", "attacker_team_code"]).sort("tick"))
pinf = pl.read_parquet(BASE / "player_info").filter(pl.col("round") == 13).select(["player_id", "team_code"])
print("round-13 roster (player_info):", pinf.group_by("team_code").len().to_dicts())
print("round-13 bomb_state rows:")
print(pl.read_parquet(BASE / "bomb_state").filter(pl.col("round") == 13).select(["tick", "event_type", "player_team_code"]))
