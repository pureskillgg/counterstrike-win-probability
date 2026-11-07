"""verify D1-6 (swarm2 verifier): schema and file-count probe for the 2024 and 2026 tomes.
Lazy scans only; prints schemas, file counts and round_end row counts. No writes.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"


def files24(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


def files26(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}*" / "dataframe_*")))


for ch in ["round_end", "round_state", "bomb_state", "player_info"]:
    f24 = files24(ch)
    f26 = files26(ch)
    print(f"[2024] {ch}: files={len(f24)}")
    if f24:
        print("   schema:", pl.read_parquet_schema(f24[0]))
    print(f"[2026] {ch}: files={len(f26)}")
    if f26:
        print("   schema:", pl.read_parquet_schema(f26[0]))

re24 = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"])
                  for f in files24("round_end")], how="vertical_relaxed")
print("[2024] round_end rows (lazy count):", re24.select(pl.len()).collect().item())
re26 = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"])
                  for f in files26("round_end")], how="vertical_relaxed")
print("[2026] round_end rows (lazy count):", re26.select(pl.len()).collect().item())
