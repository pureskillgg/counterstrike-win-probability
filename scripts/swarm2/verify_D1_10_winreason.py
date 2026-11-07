"""D1-10 verifier: internal consistency of round_end winner_team_code with the game's own win_reason.

For each tome (2026, 2024) and for the 2024 round-9/13/25/28 rows: crosstab win_reason_code vs
winner_team_code. Reason codes 9 (Terrorists_Win), 1 (Target_Bombed), 18 (CTs_Surrender) are T wins;
8 (CTs_Win), 7 (Bomb_Defused), 12 (Target_Saved), 17 (Terrorists_Surrender), 11 (All_Hostages_Rescued)
are CT wins (as read from the 2026 tome message table; re-derived here from the data, not assumed).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(12)

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")


def fl(root, ch):
    return sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))


for label, root in [("2026", T26), ("2024", T24)]:
    re_ = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message"]).collect()
                     for f in fl(root, "round_end")]).unique(["match_key", "round"])
    print(f"== {label}: round_end rows {re_.height}; win_reason x winner crosstab")
    xt = re_.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"])
    for r in xt.iter_rows(named=True):
        print(f"   reason {r['win_reason_code']!s:>3} {str(r['win_reason_message'])[:36]:<36} winner {r['winner_team_code']} n={r['len']}")
    multi = re_.group_by("win_reason_code").agg(pl.col("winner_team_code").n_unique().alias("k")).filter(pl.col("k") > 1)
    print("   reason codes with more than one winner value:", multi["win_reason_code"].to_list())
    # bomb-linked reasons: 1 (Target_Bombed) and 7 (Bomb_Defused) are the decisive ones
    bombed = re_.filter(pl.col("win_reason_code") == 1)
    defused = re_.filter(pl.col("win_reason_code") == 7)
    print(f"   Target_Bombed rows: {bombed.height}, winner != 2 (T): {bombed.filter(pl.col('winner_team_code') != 2).height}")
    print(f"   Bomb_Defused rows: {defused.height}, winner != 3 (CT): {defused.filter(pl.col('winner_team_code') != 3).height}")
    if label == "2024":
        sel = re_.filter(pl.col("round").is_in([9, 13, 25, 28]))
        print("   2024 rounds 9/13/25/28: win_reason x winner")
        print(sel.group_by(["round", "win_reason_code", "winner_team_code"]).len().sort(["round", "win_reason_code"]))
print("DONE")
