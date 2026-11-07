"""D2-7 verifier: breakdown of the kept-set flips by round number, using cross_period's own 2024 kept set
and the truth table saved by verify_D2_7_explore_r13.py (noswap score truth, bomb truth). Read-only."""
import importlib.util
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TMP = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
spec = importlib.util.spec_from_file_location("cp_kb", ROOT / "scripts" / "model" / "sufficiency" / "cross_period.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)
B = cp.run_2024()
S = B["S"]
keys = S.select(["match_key", "round", "label_ct_win"]).unique(["match_key", "round"], keep="first")
truth = pl.read_parquet(TMP / "d27_rounds_2024.parquet").select(
    ["match_key", "round", "winner", "score_winner_noswap", "bomb_bs", "maxr_probe"]) if "maxr_probe" in pl.read_parquet_schema(TMP / "d27_rounds_2024.parquet") else \
    pl.read_parquet(TMP / "d27_rounds_2024.parquet").select(["match_key", "round", "winner", "score_winner_noswap", "bomb_bs"])
J = keys.join(truth, on=["match_key", "round"], how="left")
print("kept rounds:", J.height, "| with truth row:", J.filter(pl.col("winner").is_not_null()).height)
J = J.with_columns(
    (pl.col("winner") == 3).cast(pl.Int8).alias("w_ct"),
    pl.when(pl.col("bomb_bs") == "exploded").then(pl.lit(0)).when(pl.col("bomb_bs") == "defused").then(pl.lit(1)).otherwise(pl.lit(-1)).cast(pl.Int8).alias("bomb_ct"),
    pl.when(pl.col("score_winner_noswap") == 3).then(pl.lit(1)).when(pl.col("score_winner_noswap") == 2).then(pl.lit(0)).otherwise(pl.lit(-1)).cast(pl.Int8).alias("score_ct"),
)
J = J.with_columns(pl.when(pl.col("bomb_ct") >= 0).then(pl.col("bomb_ct")).otherwise(pl.col("score_ct")).alias("truth_ct"))
fl = J.filter((pl.col("truth_ct") >= 0) & (pl.col("truth_ct") != pl.col("w_ct")))
print("\nkept-set flipped rounds (bomb first, then noswap score), by round number:")
print(fl.group_by("round").len().sort("round"))
print("flipped total:", fl.height)
print("flipped with bomb outcome:", fl.filter(pl.col("bomb_ct") >= 0).height,
      "| of which bomb == recorded winner:", fl.filter((pl.col("bomb_ct") >= 0) & (pl.col("bomb_ct") == pl.col("w_ct"))).height)
print("flipped without bomb outcome (score only):", fl.filter(pl.col("bomb_ct") < 0).height)
print("kept rounds where score truth exists and differs from winner in rounds other than 13/25/28:",
      fl.filter(~pl.col("round").is_in([13, 25, 28])).height)
