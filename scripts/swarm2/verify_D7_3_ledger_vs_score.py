"""verify D7-3: how well does the claimed ledger winner (D7 DP, w_ok / ledger_side) agree with the independent score truth,
and does the claimed match-day / folder-day labelling match match_date? Read-only; prints only.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")

S = pl.read_parquet(str(SCR / "verify_D7_3_score_rounds.parquet"))
X = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_with_ledger_dp.parquet"))
L = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_dp.parquet"))
print("ledger-rounds schema:", {k: str(v) for k, v in X.schema.items()})
print("ledger schema:", {k: str(v) for k, v in L.schema.items()})
print("X rows:", X.height, "| L rows:", L.height)

# ledger swap flags per (match, round)
fas = L.filter(pl.col("swap") == True).with_columns((pl.col("round") + 1).alias("round")) \
       .select(["match_key", "round"]).with_columns(pl.lit(True).alias("fas_ledger")).unique()

J = S.join(X.select([c for c in X.columns if c in ("match_key", "round", "ledger_side", "w_ok")])
           .unique(subset=["match_key", "round"], keep="first"),
           on=["match_key", "round"], how="left") \
     .join(fas, on=["match_key", "round"], how="left") \
     .with_columns(pl.col("fas_ledger").fill_null(False))
print("joined rows:", J.height, "| with ledger side:", J.filter(pl.col("ledger_side").is_not_null()).height)

both = J.filter(pl.col("has_score") & pl.col("ledger_side").is_not_null())
print("\nledger_side vs score winner (both available):", both.height,
      "| agree:", both.filter(pl.col("ledger_side") == pl.col("winner_score")).height)
print("\nledger_side vs score winner, by build x round bin:")
print(both.group_by(["build_num", "rbin"]).agg(
    pl.len().alias("n"),
    (pl.col("ledger_side") != pl.col("winner_score")).sum().alias("ledger_disagrees_score"),
    (pl.col("winner_team_code") != pl.col("winner_score")).sum().alias("roundend_disagrees_score"),
).sort(["build_num", "rbin"]))

print("\nledger vs score on non-round-13 rows, by build and ledger fas flag:")
nb = both.filter(pl.col("round") != 13)
print(nb.group_by(["build_num", "fas_ledger"]).agg(
    pl.len().alias("n"),
    (pl.col("ledger_side") != pl.col("winner_score")).sum().alias("ledger_disagrees_score"),
    (pl.col("winner_team_code") != pl.col("winner_score")).sum().alias("roundend_disagrees_score"),
).sort(["build_num", "fas_ledger"]))

print("\nround 13 rows, build 9961: ledger fas flag x score-vs-roundend disagreement:")
r13 = both.filter((pl.col("round") == 13) & (pl.col("build_num") == 9961))
print(r13.group_by(["fas_ledger"]).agg(
    pl.len().alias("n"),
    (pl.col("winner_team_code") != pl.col("winner_score")).sum().alias("roundend_disagrees_score"),
    (pl.col("ledger_side") != pl.col("winner_score")).sum().alias("ledger_disagrees_score"),
))

print("\n9961 round-13 rows that AGREE between round_end and score (n=201 expected): covariates")
agree13 = r13.filter(pl.col("winner_team_code") == pl.col("winner_score"))
print("n:", agree13.height)
print("by has_bomb:", agree13.group_by("has_bomb").len().rows())
print("by player_count:", agree13.group_by("player_count").len().sort("player_count").rows())
print("by map:", agree13.group_by("map_name").len().sort("len", descending=True).rows())
print("by folder day:", agree13.group_by("day").len().sort("day").rows())
print("by mdt day:", agree13.with_columns(pl.col("mdt").dt.strftime("%m-%d").alias("mdd")).group_by("mdd").len().sort("mdd").rows())
print("by reason:", agree13.group_by("win_reason_message").len().rows())

print("\nfolder day x match_date day x build (all matches, from round rows at round 1):")
Rk = S.filter(pl.col("round") == 1).select(["match_key", "day", "mdt", "build_num"]).with_columns(
    pl.col("mdt").dt.strftime("%m-%d").alias("mdd"))
print(Rk.group_by(["day", "mdd", "build_num"]).len().sort(["day", "mdd", "build_num"]).rows())
