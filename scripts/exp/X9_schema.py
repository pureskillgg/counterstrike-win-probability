import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import polars as pl
ROOT = {
 "2024": r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds",
 "2026": r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds",
}
DATES = {"2024": "2023-12-28,2024-02-25", "2026": "2026-09-18,2026-10-08"}
for era in ROOT:
    for ch in ["weapon_fire", "player_death", "player_hurt", "player_status", "round_start", "round_end", "player_info"]:
        glob = os.path.join(ROOT[era], f"basic_{ch}.{DATES[era]}", "dataframe_*")
        try:
            lf = pl.scan_parquet(glob, glob=True, missing_columns="insert", extra_columns="ignore")
            sch = lf.collect_schema()
            print(era, ch, len(sch), list(sch.keys()))
        except Exception as e:
            print(era, ch, "ERR", repr(e)[:300])
