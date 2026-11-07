"""D7-3: is the mismatch flagged by a logging signature (bomb-event tick vs round_end tick) or by
the round_end win_reason fields? Period split at the onset found in D7_2 (2024-02-07 00:00 UTC).
ASSUMPTION: player_count == 20 (2024 convention). ASSUMPTION: 64 ticks per second (header tick_rate = 64).
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
RE_wr = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_end{SUF}")]).select(
    ["match_key", "round", "win_reason_code", "win_reason_message", "legacy_code", "second"]).collect()
R = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_all.parquet"))
O = (R.filter(pl.col("bomb_side").is_not_null() & (pl.col("player_count") == 20))
       .join(RE_wr, on=["match_key", "round"], how="left")
       .with_columns(
           (pl.col("out_tick") - pl.col("tick")).alias("lag_ticks"),
           pl.when(pl.col("mdt") < pl.datetime(2024, 2, 7)).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"),
       ))
print("outcome rounds (player_count==20):", O.height, "| mismatches:", int(O["mm"].sum()))

print("\n== lag_ticks (out_tick - round_end tick) by period and mismatch ==")
print(O.group_by(["period", "mm"]).agg(pl.len().alias("n"),
      pl.col("lag_ticks").min().alias("min"), pl.col("lag_ticks").median().alias("median"),
      pl.col("lag_ticks").max().alias("max")).sort(["period", "mm"]))

print("\n== lag_ticks value counts for mismatch rows (top) ==")
print(O.filter(pl.col("mm")).group_by("lag_ticks").len().sort("len", descending=True).head(8).rows())
print("== lag_ticks value counts for clean rows, post period (top) ==")
print(O.filter((~pl.col("mm")) & (pl.col("period") == "post")).group_by("lag_ticks").len().sort("len", descending=True).head(8).rows())
print("== lag_ticks value counts for clean rows, pre period (top) ==")
print(O.filter((~pl.col("mm")) & (pl.col("period") == "pre")).group_by("lag_ticks").len().sort("len", descending=True).head(8).rows())

print("\n== crosstab: lag == -429 (mismatch signature) vs mm, all outcome rounds ==")
O2 = O.with_columns((pl.col("lag_ticks") == -429).alias("sig429"))
print(O2.group_by(["sig429", "mm"]).len().sort(["sig429", "mm"]))
print("== same, restricted to rounds 13 and 25 ==")
print(O2.filter(pl.col("rbin") != "other").group_by(["rbin", "sig429", "mm"]).len().sort(["rbin", "sig429", "mm"]))

print("\n== win_reason_message x bomb outcome x mm (all outcome rounds) ==")
O = O.with_columns(pl.when(pl.col("bomb_side") == 3).then(pl.lit("defused")).otherwise(pl.lit("exploded")).alias("outcome"))
print(O.group_by(["win_reason_message", "outcome", "winner_team_code", "mm"]).len().sort(["win_reason_message", "outcome", "winner_team_code", "mm"]))

print("\n== win_reason_code x mm (outcome rounds) ==")
print(O.group_by(["win_reason_code", "mm"]).len().sort(["win_reason_code", "mm"]))

print("\n== the 7 post-onset clean r13 rows ==")
exc = O.filter((pl.col("round") == 13) & (pl.col("period") == "post") & (~pl.col("mm")))
print(exc.select(["mdt", "match_key", "map_name", "winner_team_code", "bomb_side", "planter_side", "lag_ticks",
                  "win_reason_message", "round"]).sort("mdt").with_columns(pl.col("match_key").str.slice(0, 36)))

print("\n== round 9 mismatches (the non-swap 'other' round) ==")
r9 = O.filter((pl.col("round") == 9) & pl.col("mm"))
print(r9.select(["mdt", "match_key", "map_name", "winner_team_code", "bomb_side", "planter_side", "lag_ticks",
                 "win_reason_message"]).sort("mdt").with_columns(pl.col("match_key").str.slice(0, 36)))
print("round-9 outcome rows overall, by mm:", O.filter(pl.col("round") == 9).group_by("mm").len().rows())

print("\n== round 28 mismatches (OT round after swap) ==")
print(O.filter((pl.col("round") == 28) & pl.col("mm")).select(["mdt", "map_name", "winner_team_code", "bomb_side", "lag_ticks", "win_reason_message"]).sort("mdt"))
