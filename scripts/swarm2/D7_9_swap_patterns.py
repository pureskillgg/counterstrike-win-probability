"""D7-9: per-match swap patterns from the min-swap ledger path (D7_8), and checks on format and disagreements."""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
L = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_dp.parquet"))
X = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_with_ledger_dp.parquet"))
sw = (L.filter(pl.col("swap") == True).group_by("match_key").agg(pl.col("round").sort().alias("swap_rounds")))
mx = X.group_by("match_key").agg(pl.col("round").max().alias("maxr"), pl.col("mdt").first(), pl.col("map_name").first())
M = mx.join(sw, on="match_key", how="left").with_columns(pl.col("swap_rounds").fill_null([]).alias("swap_rounds"))
M = M.with_columns(pl.col("swap_rounds").list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("pattern"))
print("swap patterns (matches), top 15:")
print(M.group_by("pattern").len().sort("len", descending=True).head(15).rows())
print("\nmatches with a swap at round 8 - max-round distribution:")
print(M.filter(pl.col("swap_rounds").list.contains(8)).group_by("maxr").len().sort("maxr").rows())
print("matches with swap at round 12 - max-round distribution (top 8):")
print(M.filter(pl.col("swap_rounds").list.contains(12)).group_by("maxr").len().sort("len", descending=True).head(8).rows())
print("\npattern vs maxr (top 15):")
print(M.group_by(["pattern", "maxr"]).len().sort("len", descending=True).head(15).rows())

# disagreement check by pattern
D = X.join(M.select(["match_key", "pattern", "maxr"]), on="match_key", how="left")
print("\nwinner-vs-ledger disagreement by pattern (rows with a unique ledger side):")
print(D.filter(pl.col("w_ok").is_not_null()).group_by(["pattern", "period", "w_ok"]).len().sort("len", descending=True).head(12).rows())
print("\npre-onset disagreements (51 in DP version) - by pattern and round:")
pre_d = D.filter((pl.col("period") == "pre") & (pl.col("w_ok") == False))
print(pre_d.group_by(["pattern", "round"]).len().sort("len", descending=True).head(12).rows())
print(pre_d.select(["mdt", "match_key", "round", "winner_team_code", "ledger_side", "bomb_side", "map_name"]).sort("mdt").head(12).with_columns(pl.col("match_key").str.slice(0,40)))
