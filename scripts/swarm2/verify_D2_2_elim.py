"""Verifier D2-2: which truth source produces the flips outside rounds 13/25/28 in the recompute build?

Imports build() from verify_D2_2_recompute.py (read-only use) and tabulates flags by truth source.
Run: G:\pureskill\csds-cs2\.pixi\envs\default\python.exe -I -X utf8 scripts\swarm2\verify_D2_2_elim.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2")))
sys.stdout.reconfigure(encoding="utf-8")
import polars as pl  # noqa: E402

import verify_D2_2_recompute as rc  # noqa: E402

B, F = rc.build("2024")
B = B.filter(pl.col("player_count") == 20).filter(pl.col("T").is_not_null())
B = B.with_columns(
    pl.when(pl.col("s_score").is_not_null()).then(pl.lit("score"))
    .when(pl.col("s_bomb").is_not_null()).then(pl.lit("bomb"))
    .otherwise(pl.lit("elim_only")).alias("src"))
print("rows with T:", B.height, " flagged:", B.filter(pl.col("flag")).height)
print("flagged by source and round bin:")
print(B.filter(pl.col("flag")).group_by(["src", "bin"]).len().sort(["src", "bin"]))
oth = B.filter(pl.col("flag") & ~pl.col("round").is_in([13, 25, 28]))
print("flagged rows outside 13/25/28 by source:", oth.group_by("src").len().sort("src").rows())
print("R != s_elim where s_elim is the only truth, by bin (rows, disagree):")
eo = B.filter(pl.col("src") == "elim_only")
print(eo.group_by("bin").agg(pl.len().alias("n"), pl.col("flag").sum().alias("disagree")).sort("bin").rows())
print("R != s_elim where score or bomb exist, by bin (rows, disagree):")
so = B.filter(pl.col("src") != "elim_only")
print(so.group_by("bin").agg(pl.len().alias("n"), pl.col("flag").sum().alias("disagree")).sort("bin").rows())
print("elim_only flags outside 13/25/28, with planter present (bomb planted):",
      oth.filter((pl.col("src") == "elim_only") & pl.col("planter_code").is_not_null()).height)
