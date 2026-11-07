"""D6-4 verifier: win_reason_code -> message map and side, both eras (compact)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
def rend(base):
    fd = [p for p in base.iterdir() if p.name.startswith("basic_round_end")][0]
    return pl.scan_parquet(str(fd / "dataframe_*")).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"]).collect()
for name, base in [("2024", T24), ("2026", T26)]:
    d = rend(base)
    print(name, "rows", d.height)
    print(d.group_by(["win_reason_code", "win_reason_message"]).agg(
        pl.len().alias("n"), pl.col("winner_team_code").value_counts().alias("winner_counts")).sort("win_reason_code"))
