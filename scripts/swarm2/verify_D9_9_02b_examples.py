"""verify_D9_9 step 2b (exploration): per-match sequences around the halftime and OT rounds.

Prints round-by-round: round_end winner/message/tick, bomb outcome, round_officially_ended tick and scores,
score deltas, roster counts and death counts. Reads scratchpad tables only.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
R = pl.read_parquet(SCR / "verify_D9_9_round.parquet")
M = pl.read_parquet(SCR / "verify_D9_9_match.parquet")
cols = ["round", "tick", "winner_team_code", "win_reason_message", "bomb_side", "ro_tick", "t_after", "ct_after",
        "dT_straight", "dCT_straight", "dT_swapped", "dCT_swapped", "n_T", "n_CT", "T_dead", "CT_dead"]
for b in [9961, 9842]:
    keys = M.filter(pl.col("build_num") == b).select("match_key").head(2)["match_key"].to_list()
    for k in keys:
        print(f"\n### build {b} match {k}")
        print(R.filter(pl.col("match_key") == k).sort("round").select(cols).head(16))
