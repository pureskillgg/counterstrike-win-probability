"""A10 schema probe (read-only). Prints column names and row counts of raw E: channels for the
anomaly match, so the round-16 probe uses the same column names as the extractor."""
import os
import sys

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

MATCH_DIR = r"E:\csds_final\csds\2026\10\07\22e9c3e7-a333-4ebc-a51c-19ebbcc13ef1"
CHANNELS = ["player_status", "player_vector", "player_death", "player_info", "round_state",
            "round_start", "round_end", "team_change", "player_disconnect", "bot_takeover",
            "bomb_state", "header", "tick"]

for ch in CHANNELS:
    path = os.path.join(MATCH_DIR, ch)
    if not os.path.exists(path):
        print(f"{ch}: MISSING")
        continue
    ds = pq.ParquetDataset(path)
    names = ds.schema.names
    n = sum(f.count_rows() for f in ds.fragments)
    print(f"{ch}: rows={n} cols={names}")
