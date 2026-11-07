"""verify D2-6: compare D2's intermediate round table (sx, winner, pc) with the raw-derived values to locate the population gap."""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
pl.Config.set_tbl_rows(30)
for label in ["2024"]:
    d2 = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_rounds_{label}.parquet")
    print("D2_rounds columns:", d2.columns)
    print("rows", d2.height, "matches", d2["match_key"].n_unique())
    print(d2.head(3))
    if "pc" in d2.columns or "player_count" in d2.columns:
        pass
    for c in ["sx", "winner", "pc", "player_count"]:
        if c in d2.columns:
            print(c, "nulls:", d2[c].null_count(), "distinct:", d2[c].n_unique())
