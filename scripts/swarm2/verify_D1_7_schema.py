"""verify D1-7 (swarm2 verifier): schema probe for the tables used by the D1-7 checks."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"

print("polars", pl.__version__)
for name, base, pat in [
    ("2024 round_end", T24, f"basic_round_end{SUF24}"),
    ("2024 round_state", T24, f"basic_round_state{SUF24}"),
    ("2024 bomb_state", T24, f"basic_bomb_state{SUF24}"),
    ("2026 round_end", T26, "basic_round_end*"),
    ("2026 round_state", T26, "basic_round_state*"),
    ("2026 bomb_state", T26, "basic_bomb_state*"),
    ("2026 score_update", T26, "basic_score_update*"),
]:
    files = sorted(glob.glob(str(base / pat / "dataframe_*")))
    print(f"\n== {name}: {len(files)} dataframe files")
    if not files:
        continue
    lf = pl.scan_parquet(files)
    sch = lf.collect_schema()
    print({k: str(v) for k, v in sch.items()})
    print("rows:", lf.select(pl.len()).collect().item())
