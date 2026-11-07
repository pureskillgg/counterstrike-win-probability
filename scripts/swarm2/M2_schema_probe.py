"""M2 probe: column schemas and key value counts for the round-level channels in both tomes."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
ROOT = Path("G:/pureskill/csds-cs2")
ERAS = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}
CHANS = ["round_end", "round_start", "round_state", "player_info", "player_death", "bomb_state",
         "player_disconnect", "bot_takeover", "team_change"]

for era, (base, suf) in ERAS.items():
    print(f"===== {era} =====")
    for ch in CHANS:
        fd = base / f"basic_{ch}{suf}"
        if not fd.exists():
            print(f"  {ch}: MISSING FOLDER")
            continue
        files = sorted(glob.glob(str(fd / "dataframe_*")))
        keyfiles = sorted(glob.glob(str(fd / "keyset_*")))
        sch = pl.read_parquet_schema(files[0])
        print(f"  {ch}: pages={len(files)} keyset_pages={len(keyfiles)} cols={list(sch.items())}")
    # round_end player_count distribution and winner codes
    fd = base / f"basic_round_end{suf}"
    re_ = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "win_reason_code",
                                                  "player_count", "tick"])
                     for f in sorted(glob.glob(str(fd / "dataframe_*")))], how="vertical_relaxed")
    print(f"  round_end rows={re_.height} matches={re_['match_key'].n_unique()}")
    print("  player_count:", re_.group_by("player_count").agg(pl.len()).sort("player_count").rows())
    print("  winner_team_code:", re_.group_by("winner_team_code").agg(pl.len()).sort("winner_team_code").rows())
    print("  win_reason_code:", re_.group_by("win_reason_code").agg(pl.len()).sort("win_reason_code").rows())
    fd = base / f"basic_round_state{suf}"
    rs = pl.concat([pl.read_parquet(f, columns=["event_type"])
                    for f in sorted(glob.glob(str(fd / "dataframe_*")))], how="vertical_relaxed")
    print("  round_state event_type:", rs.group_by("event_type").agg(pl.len()).sort("event_type").rows())
    print()
