"""verify D9-8 step 13: freeze-to-freeze consistency on the 2026 tome (bomb-free), same rule as step 12.

Reads round_end (player_count == 10, the 2026 standard round) and round_state (round_freeze_end phase and
scores) directly from the 2026 tome, projected columns only. Rule: bake-off swap_expected with max_rounds
inferred from round_state phase (ASSUMPTION A4, as in step 12). Reports inconsistent transitions by round.
Output goes to stdout (tee'd by the caller). Scratch parquet only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUF = ".2026-09-18,2026-10-08"


def files_of(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


re26 = (pl.scan_parquet(files_of("round_end")).select(["match_key", "round", "winner_team_code", "player_count"])
        .filter(pl.col("player_count") == 10).group_by(["match_key", "round"])
        .agg(pl.col("winner_team_code").first()).collect())
fz = (pl.scan_parquet(files_of("round_state")).select(["match_key", "round", "event_type", "t_score", "ct_score", "phase"])
      .filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"])
      .agg(pl.col("t_score").first(), pl.col("ct_score").first(), pl.col("phase").first()).collect())
print("2026 pc10 round_end (match, round):", re26.height, "  freeze_end (match, round):", fz.height)
print("phase values at freeze_end:", fz.group_by("phase").agg(pl.len().alias("n")).sort("phase").rows())

ht = fz.filter(pl.col("phase") == "second").group_by("match_key").agg((pl.col("round").min() - 1).alias("H"))
print("H distribution:", ht.group_by("H").agg(pl.len().alias("matches")).sort("H").rows())
ht = ht.with_columns((2 * pl.col("H")).alias("max_rounds"))
fz2 = fz.join(ht, on="match_key", how="left")
nxt = fz2.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("t_score").alias("t_next"), pl.col("ct_score").alias("ct_next")])
T = fz2.join(re26, on=["match_key", "round"], how="inner").join(nxt, on=["match_key", "round"], how="inner")
T = T.with_columns(
    pl.when(pl.col("max_rounds").is_null()).then(pl.lit(False))
    .when(pl.col("round") == pl.col("H")).then(pl.lit(True))
    .when((pl.col("round") > pl.col("max_rounds")) & (((pl.col("round") - pl.col("max_rounds") - 3) % 6) == 0)).then(pl.lit(True))
    .otherwise(pl.lit(False)).alias("swap"))
T = T.with_columns(
    (pl.col("t_score") + (pl.col("winner_team_code") == 2).cast(pl.Int64)).alias("t_post"),
    (pl.col("ct_score") + (pl.col("winner_team_code") == 3).cast(pl.Int64)).alias("ct_post"))
T = T.with_columns(
    pl.when(pl.col("swap")).then(pl.col("ct_post")).otherwise(pl.col("t_post")).alias("pred_t"),
    pl.when(pl.col("swap")).then(pl.col("t_post")).otherwise(pl.col("ct_post")).alias("pred_ct"))
T = T.with_columns(((pl.col("pred_t") == pl.col("t_next")) & (pl.col("pred_ct") == pl.col("ct_next"))).alias("consistent"))
print(f"\n2026 transitions: {T.height}; consistent: {int(T['consistent'].sum())}; inconsistent: {int((~T['consistent']).sum())}")
inc = T.filter(~pl.col("consistent"))
print("inconsistent by round:")
print(inc.group_by("round").agg(pl.len().alias("n")).sort("round").head(40))
print("max_rounds distribution among transitions:", T.group_by("max_rounds").agg(pl.len().alias("n")).sort("max_rounds").rows())
print("swap-round transitions checked:", int(T["swap"].sum()), " consistent:", int((T["swap"] & T["consistent"]).sum()))
