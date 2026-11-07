# verify A1-2: is round_end.winner_team_code a side code with 2=T, 3=CT?
# Ground truth: win_reason_message text (names the winning side), cross-tabbed with winner_team_code, over all 2026 matches.
import glob, os
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
def pages(ch):
    return sorted(glob.glob(os.path.join(T, f"basic_{ch}.2026-09-18,2026-10-08", "dataframe_*")))
re_ = pl.concat([pl.scan_parquet(p).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message"]) for p in pages("round_end")], how="diagonal_relaxed").collect()
print("round_end rows", re_.height, "| matches", re_.get_column("match_key").n_unique())
print("crosstab (win_reason_message, winner_team_code):")
print(re_.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]).rows())
print("duplicate (match_key, round) rows in round_end:", re_.height - re_.unique(["match_key", "round"]).height)
