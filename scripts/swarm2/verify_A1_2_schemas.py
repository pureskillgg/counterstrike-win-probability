# verify A1-2: schemas of channels that could carry per-player side / team identity (read-only, lazy scan).
import os, glob
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
want = ["basic_round_start", "basic_round_state", "basic_player_info", "basic_team_change",
        "basic_score_update", "basic_player_spawn", "basic_player_death", "basic_bomb_state",
        "basic_player_personal", "basic_round_mvp"]
for c in want:
    folder = [d for d in os.listdir(T) if d.startswith(c + ".")]
    if not folder:
        print(c, "NOT FOUND")
        continue
    files = sorted(glob.glob(os.path.join(T, folder[0], "dataframe_*")))
    sch = pl.scan_parquet(files).collect_schema()
    print(f"{c}: {len(files)} files")
    print("   ", ", ".join(f"{n}:{str(t)}" for n, t in sch.items()))
