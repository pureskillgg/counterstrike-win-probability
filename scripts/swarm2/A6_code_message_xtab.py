import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(120)
ROOTS = {
    "2024": Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds") / "basic_round_end.2023-12-28,2024-02-25",
    "2026": Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds") / "basic_round_end.2026-09-18,2026-10-08",
}
for era, fd in ROOTS.items():
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    cols = ["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"]
    if era == "2024":
        cols.append("legacy_code")
    lf = pl.scan_parquet(files).select(cols)
    df = lf.collect()
    print(f"==== {era}: rows={df.height}, matches={df['match_key'].n_unique()}")
    print("null counts:", {c: df[c].null_count() for c in df.columns})
    print("winner_team_code counts:")
    print(df.group_by("winner_team_code", maintain_order=True).len().sort("winner_team_code"))
    print("win_reason_code x winner_team_code:")
    print(df.group_by(["win_reason_code", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))
    print("win_reason_code x win_reason_message:")
    print(df.group_by(["win_reason_code", "win_reason_message"]).len().sort(["win_reason_code", "win_reason_message"]))
    if era == "2024":
        print("legacy_code counts:")
        print(df.group_by("legacy_code").len().sort("legacy_code"))
        print("legacy_code x win_reason_code:")
        print(df.group_by(["legacy_code", "win_reason_code"]).len().sort(["legacy_code", "win_reason_code"]))
        print("legacy_code x winner_team_code:")
        print(df.group_by(["legacy_code", "winner_team_code"]).len().sort(["legacy_code", "winner_team_code"]))
    print("player_count counts:")
    print(df.group_by("player_count").len().sort("player_count"))
