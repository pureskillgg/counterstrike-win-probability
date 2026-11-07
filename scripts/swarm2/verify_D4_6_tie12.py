"""Control for the cross-swap tie theory at the halftime boundary (12 -> 13) and the OT boundary (24 -> 25).
For rounds 12 and 24 (days 12-25): tie status at freeze r, whether S is valid, and S vs W among valid rows.
Also: rounds 9, 13, 25, 28 (no swap at r -> r+1): S vs W among valid rows, by tie status, as a contrast.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
X = R.filter((pl.col("dd") >= 12) & (pl.col("player_count") == 20)).with_columns(
    (pl.col("t_score") == pl.col("ct_score")).alias("tied"))
for r in [9, 12, 13, 24, 25, 27, 28]:
    Y = X.filter(pl.col("round") == r)
    out = Y.group_by("tied").agg(pl.len().alias("n"), (pl.col("S") != 0).sum().alias("n_S_valid"),
                                 ((pl.col("S") != 0) & (pl.col("S") != pl.col("W"))).sum().alias("n_valid_S_ne_W"),
                                 ((pl.col("S") != 0) & (pl.col("S") == pl.col("W"))).sum().alias("n_valid_S_eq_W")).sort("tied")
    print(f"round {r}:")
    print(out)
