"""verify D9-8 step 2: recompute the round-13 mismatch numbers from the 2024 tome.

Definitions (mirroring scripts\\swarm\\e2_side_check_r13.py, re-implemented here, not imported):
  - round rows: round_end, player_count == 20 (ASSUMPTION: 20 = full 10v10 rounds; the claim uses the same filter)
  - bomb outcome: bomb_state event_type in {bomb_defused, bomb_exploded}, per (match_key, round)
  - bomb_side: 3 (CT) if any bomb_defused row in the round, else 2 (T) if bomb_exploded
  - mismatch: bomb_side != winner_team_code
  - generic message: win_reason_message in {Terrorists_Win, CTs_Win}; specific: Bomb_Defused, Target_Bombed
  - day window: match_key path csds/2024/02/DD/ ; defect window = days 12-25, clean = 01-07 (as in the claim)

Only projected columns are read. Output: stdout (tee'd into logs\\swarm2\\verify_D9_8.log by the caller).
Intermediate parquet goes to the scratchpad, not to the project.
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)


def files_of(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def lazy(ch, cols):
    return pl.scan_parquet(files_of(ch)).select(cols)


GENERIC = ["#SFUI_Notice_Terrorists_Win", "#SFUI_Notice_CTs_Win"]
SPECIFIC = ["#SFUI_Notice_Bomb_Defused", "#SFUI_Notice_Target_Bombed"]

re_all = (
    lazy("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                       "win_reason_message", "player_count"])
    .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
    .collect()
)
print(f"round_end rows total: {re_all.height}")
print("day values present (round_end):", sorted(re_all["day"].unique().to_list()))

# duplicate and gap checks on round numbering per match
dup = (re_all.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
       .filter(pl.col("n") > 1))
print(f"\n(match, round) pairs with >1 round_end row: {dup.height}")
if dup.height:
    print(dup.group_by("n").agg(pl.len().alias("pairs")).sort("n"))
    print(dup.head(10))

rs_all = (
    lazy("round_start", ["match_key", "round", "tick"]).collect()
)
print(f"round_start rows: {rs_all.height}")
dup_rs = rs_all.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
print(f"(match, round) pairs with >1 round_start row: {dup_rs.height}")

bs_all = (
    lazy("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
)
print(f"bomb_state rows: {bs_all.height}")

# ------- outcome per (match, round)
outc = (
    bs_all.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
    .group_by(["match_key", "round"])
    .agg(
        pl.col("event_type").unique().alias("evs"),
        pl.col("tick").filter(pl.col("event_type") == "bomb_defused").min().alias("defuse_tick"),
        pl.col("tick").filter(pl.col("event_type") == "bomb_exploded").min().alias("explode_tick"),
        pl.len().alias("n_outcome_rows"),
    )
    .with_columns(
        pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
        pl.col("evs").list.contains("bomb_defused").alias("has_defused"),
        pl.col("evs").list.contains("bomb_exploded").alias("has_exploded"),
    )
)
print(f"\n(match, round) pairs with a bomb outcome event: {outc.height}")
print("pairs with BOTH defused and exploded events:",
      int((outc["has_defused"] & outc["has_exploded"]).sum()))

plant = (
    bs_all.filter(pl.col("event_type") == "bomb_planted")
    .group_by(["match_key", "round"])
    .agg(pl.col("tick").min().alias("plant_tick"), pl.col("player_team_code").first().alias("planter_side"))
)

# ------- join round_end (player_count 20) with outcome
rr = (
    re_all.filter(pl.col("player_count") == 20)
    .join(outc, on=["match_key", "round"], how="inner")
    .join(plant, on=["match_key", "round"], how="left")
    .with_columns(
        (pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
        pl.col("win_reason_message").is_in(GENERIC).alias("generic"),
        pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dgrp"),
        pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rbin"),
    )
)
print(f"\nround_end (player_count=20) rows with a bomb outcome: {rr.height}")
print("day coverage of joined rows:", sorted(rr["day"].unique().to_list()))
print("mismatch total:", int(rr["mismatch"].sum()))

print("\n== A. mismatch by round bin x day group")
print(rr.group_by(["dgrp", "rbin"]).agg(
    pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
    pl.col("mismatch").mean().round(4).alias("rate")).sort(["dgrp", "rbin"]))

print("\n== B. message type x mismatch, round 13 only (both day groups)")
r13 = rr.filter(pl.col("round") == 13)
print(r13.group_by(["dgrp", "mismatch", "win_reason_message", "winner_team_code", "bomb_side"]).agg(
    pl.len().alias("n")).sort(["dgrp", "mismatch", "win_reason_message"]))

print("\n== C. generic vs specific x mismatch, by group")
print(rr.group_by(["dgrp", "rbin", "generic", "mismatch"]).agg(pl.len().alias("n")).sort(
    ["dgrp", "rbin", "generic", "mismatch"]))

# headline numbers
m13 = r13.filter(pl.col("mismatch"))
print(f"\nround 13 mismatch rows: {m13.height}  of which generic: {int(m13['generic'].sum())}  "
      f"specific: {int((~m13['generic']).sum())}")
print("round 13 mismatch rows by day group:", m13.group_by("dgrp").agg(pl.len().alias("n")).sort("dgrp").rows())
m13d = r13.filter(pl.col("dgrp") == "d12-25")
print(f"round 13, d12-25: total bomb rows {m13d.height}, mismatch {int(m13d['mismatch'].sum())}, "
      f"rate {m13d['mismatch'].mean():.4f}")
mt = r13.filter(~pl.col("mismatch"))
print(f"round 13 matched rows total: {mt.height}  specific {int((~mt['generic']).sum())}  "
      f"generic {int(mt['generic'].sum())}")
for dg in ["d01-07", "d12-25"]:
    sub = mt.filter(pl.col("dgrp") == dg)
    print(f"  matched in {dg}: n={sub.height}  specific={int((~sub['generic']).sum())}  "
          f"generic={int(sub['generic'].sum())}  "
          f"msg counts={sub.group_by('win_reason_message').agg(pl.len().alias('n')).sort('win_reason_message').rows()}")

rr.write_parquet(SCRATCH / "verify_D9_8_rounds.parquet")
bs_all.write_parquet(SCRATCH / "verify_D9_8_bomb_state.parquet")
re_all.write_parquet(SCRATCH / "verify_D9_8_round_end.parquet")
rs_all.write_parquet(SCRATCH / "verify_D9_8_round_start.parquet")
print(f"\nwrote scratch parquet to {SCRATCH}")
