"""verify_D2_6 probe 5: raw vs de-duplicated player_info counts per round, 2026 and 2024. Prints only."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds", "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
for label in ["2026", "2024"]:
    fd = sorted(glob.glob(str(TOME[label] / "basic_player_info.*")))[0]
    pi = pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(["match_key", "round", "player_id", "team_code"]).collect()
    raw = pi.group_by(["match_key", "round"]).agg(pl.len().alias("n_raw"), pl.col("player_id").n_unique().alias("n_ids"),
                                                 (pl.col("team_code") == 2).sum().alias("nT_raw"), (pl.col("team_code") == 3).sum().alias("nCT_raw"))
    print(label, "rounds in player_info:", raw.height, " rounds with duplicate player rows:", raw.filter(pl.col("n_raw") > pl.col("n_ids")).height)
    print(label, " raw valid (np==10, 5/5):", raw.filter((pl.col("n_raw") == 10) & (pl.col("nT_raw") == 5) & (pl.col("nCT_raw") == 5)).height,
          " dedup valid (ids==10, 5/5):", raw.filter((pl.col("n_ids") == 10) & (pl.col("nT_raw") == 5) & (pl.col("nCT_raw") == 5)).height)
