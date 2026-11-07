"""verify D9-9, step 6: probe final-round score rows and swap boundaries by round index (read-only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9_9verify")
RS = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect()
H = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "build_num", "t_starters_score_final", "ct_starters_score_final"]).collect().rename({"key": "match_key"})
J = pl.read_parquet(SCR / "round_score_join.parquet")

last = J.group_by("match_key").agg(pl.col("round").max().alias("last_round"))
k = "csds/2024/02/12/01648186-3840-41d9-8090-4bea9e926412/csds"
print("example last round rows, all events after round 12:")
print(RS.filter((pl.col("match_key") == k) & (pl.col("round") >= 16)).sort("tick").select(
    ["round", "tick", "event_type", "t_score", "ct_score"]).head(30))
print("example header:", H.filter(pl.col("match_key") == k).rows())

cwp = RS.filter(pl.col("event_type") == "cs_win_panel_match").select(
    ["match_key", "tick", "t_score", "ct_score"]).unique(subset=["match_key"], keep="last")
print("\ncs_win_panel_match rows:", cwp.height, "| matches:", J.select("match_key").n_unique())
X = cwp.join(H, on="match_key", how="left")
print("cs_win_panel_match score equals header finals (as-is):",
      X.filter((pl.col("t_score") == pl.col("t_starters_score_final")) &
               (pl.col("ct_score") == pl.col("ct_starters_score_final"))).height,
      "| swapped:", X.filter((pl.col("ct_score") == pl.col("t_starters_score_final")) &
                             (pl.col("t_score") == pl.col("ct_starters_score_final"))).height,
      "| of", X.height)
print("sample:", X.head(6).rows())

# boundary swap status by round index, non-tie boundaries only (uses pre(r) and post(r-1) from J)
Q = J.sort(["match_key", "round"]).with_columns(
    pl.col("t_post").shift(1).over("match_key").alias("pt"),
    pl.col("ct_post").shift(1).over("match_key").alias("pc"),
    pl.col("round").shift(1).over("match_key").alias("pr"))
Q = Q.filter((pl.col("pr") == pl.col("round") - 1) & pl.col("t_pre").is_not_null() & pl.col("pt").is_not_null())
Q = Q.with_columns(
    ((pl.col("t_pre") == pl.col("pt")) & (pl.col("ct_pre") == pl.col("pc"))).alias("noswap"),
    ((pl.col("t_pre") == pl.col("pc")) & (pl.col("ct_pre") == pl.col("pt"))).alias("swap"),
)
Q = Q.with_columns(
    pl.when(pl.col("noswap") & ~pl.col("swap")).then(pl.lit("noswap"))
      .when(pl.col("swap") & ~pl.col("noswap")).then(pl.lit("swap"))
      .when(pl.col("noswap") & pl.col("swap")).then(pl.lit("tie"))
      .otherwise(pl.lit("neither")).alias("bstat"))
print("\nboundary status (into round r) by r, build 9961, rounds 2-30 (count per status):")
Z = Q.filter(pl.col("build_num") == 9961).group_by(["round", "bstat"]).len().sort(["round", "bstat"])
piv = Z.pivot(on="bstat", index="round", values="len").sort("round").fill_null(0)
print(piv.filter(pl.col("round") <= 30))
print("\nboundary status (into round r) by r, build 9842, rounds 2-30:")
Z = Q.filter(pl.col("build_num") == 9842).group_by(["round", "bstat"]).len().sort(["round", "bstat"])
piv = Z.pivot(on="bstat", index="round", values="len").sort("round").fill_null(0)
print(piv.filter(pl.col("round") <= 30))
