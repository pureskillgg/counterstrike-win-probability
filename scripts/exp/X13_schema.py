"""X13 schema probe: column names/dtypes and event types for the channels used by X13."""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import polars as pl

ROOT = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
DATE = "2026-09-18,2026-10-08"
CHANNELS = ["bomb_action", "bomb_state", "bomb_defuse", "molotov_state", "molotov_fire",
            "player_hurt", "player_info", "round_end", "round_state", "player_death"]

for ch in CHANNELS:
    glob = os.path.join(ROOT, f"basic_{ch}.{DATE}", "dataframe_*")
    lf = pl.scan_parquet(glob)
    sch = lf.collect_schema()
    n = lf.select(pl.len()).collect().item()
    print(f"== {ch}: rows={n}")
    print("   cols:", ", ".join(f"{k}:{v}" for k, v in sch.items()))
    if "event_type" in sch:
        et = lf.group_by("event_type").agg(pl.len().alias("n")).sort("n", descending=True).collect()
        print("   event_type:", et.rows())
