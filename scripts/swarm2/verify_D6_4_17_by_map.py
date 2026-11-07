"""D6-4 verifier, step 17: does the round-13 inversion depend on map? (2024, player_count 20, days 12-25)
Score-based (side-attached, no swap) disagreement with round_end, split by header map_name. Counts only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
hdr = (pl.scan_parquet(str(ROOT / "tomes" / "2024-02-g" / "tome" / "csds" / "header" / "dataframe_*"))
       .select(["match_id", "map_name"]).collect().unique(subset=["match_id"]))
j = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet")).filter(pl.col("player_count") == 20)
j = j.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/\d{2}/([^/]+)/", 1).alias("match_id"))
j = j.join(hdr, on="match_id", how="left").filter(pl.col("day").is_in([f"{x:02d}" for x in range(12, 26)]))
print("join coverage (rows with map):", j.filter(pl.col("map_name").is_not_null()).height, "of", j.height)
r13 = j.filter((pl.col("round") == 13) & pl.col("score_side").is_not_null()).with_columns(
    (pl.col("score_side") != pl.col("winner_team_code")).alias("dis"))
print(r13.group_by("map_name").agg(pl.len().alias("n"), pl.col("dis").sum().alias("disagree"),
                                    (pl.col("dis").mean() * 100).round(1).alias("pct")).sort("n", descending=True).head(12))
print("map is NOT a clean split if all maps show the same share; compare the pct column.")
