"""verify_D9_10 step 11: reconcile the defect counts quoted in the task (434 of 605 r13; 41 of 14,602 other rounds)
and test whether win_reason_message is an independent winner check on the inverted rounds.
ASSUMPTION: population = pc20 rounds with a bomb outcome; date window = match day 12..25 (2024-02-12..25).
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_10")
R = pl.read_parquet(SCR / "round_table.parquet").with_columns(
    pl.col("day").cast(pl.Int64).alias("dayn"))
B = R.filter(pl.col("has_bomb_outcome")).with_columns(
    pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
    (pl.col("winner_team_code") != pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2))).alias("mismatch"))
P = B.filter(pl.col("player_count") == 20)
print("pc20 bomb-outcome rounds, all dates:", P.height, "| mismatches:", int(P["mismatch"].sum()))
w = P.filter(pl.col("dayn").is_between(12, 25))
r13 = w.filter(pl.col("round") == 13)
oth = w.filter(pl.col("round") != 13)
print("pc20, days 12-25: round 13 rows:", r13.height, "mismatch:", int(r13["mismatch"].sum()),
      "| other rounds rows:", oth.height, "mismatch:", int(oth["mismatch"].sum()))
print("pc20, days 12-25, round 13 by build:", r13.group_by("build_num").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort("build_num").rows())
print("pc20 all dates, round 13 rows:", P.filter(pl.col("round") == 13).height,
      "mismatch:", int(P.filter(pl.col("round") == 13)["mismatch"].sum()))
print("pc20 all dates, other rounds with mismatch by round:",
      P.filter(pl.col("round") != 13).filter(pl.col("mismatch")).group_by(["build_num", "round"]).len().sort(["build_num", "round"]).rows())
print("NOTE: claim quotes 'other rounds 41 of 14,602' and '434 of 605'; the pc20 totals above are the reference.")

# reason-message independence on mismatched rounds
M = P.filter(pl.col("mismatch"))
print("\nmismatched pc20 rounds: win_reason_message x bomb outcome")
print(M.group_by(["bomb_side", "winner_team_code", "win_reason_message"]).len().sort(["bomb_side", "winner_team_code"]))
