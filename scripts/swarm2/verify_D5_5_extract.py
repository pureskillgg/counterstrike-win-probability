"""verify D5-5, step 1: independent extraction from the raw tomes (2024 and 2026).

Reads only the channels needed, with predicate pushdown and column projection, one parquet
file at a time. Writes small intermediates to the scratchpad (not the project, not E:):
  header / round_end (all rounds) / freeze_end scores (round_state) / bomb outcome rows (bomb_state)

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_5_extract.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

TOMES = {
    "2024": Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"),
    "2026": Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"),
}
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")
SCR.mkdir(parents=True, exist_ok=True)


def channel_dir(tome, ch):
    hits = sorted(p for p in tome.glob(f"basic_{ch}.*") if p.is_dir())
    if len(hits) != 1:
        raise RuntimeError(f"expected one folder for {ch} in {tome}, got {hits}")
    return hits[0]


def scan_channel(tome, ch, cols, where=None):
    """Concatenate one channel across its parquet files, projecting and filtering lazily."""
    files = sorted(glob.glob(str(channel_dir(tome, ch) / "dataframe_*")))
    if not files:
        raise RuntimeError(f"no dataframe files for {ch}")
    frames = []
    for f in files:
        lf = pl.scan_parquet(f)
        names = lf.collect_schema().names()
        keep = [c for c in cols if c in names]
        missing = [c for c in cols if c not in names]
        if missing:
            raise RuntimeError(f"{ch} missing columns {missing} in {f}")
        lf = lf.select(keep)
        if where is not None:
            lf = lf.filter(where)
        frames.append(lf.collect())
    return pl.concat(frames, how="vertical_relaxed")


def header(tome):
    f = sorted(glob.glob(str(tome / "header" / "dataframe_*")))[0]
    return pl.read_parquet(f, columns=["key", "match_id", "map_name", "match_date", "tick_rate",
                                       "t_starters_score_final", "ct_starters_score_final"])


for year, tome in TOMES.items():
    print(f"\n===== {year} =====")
    H = header(tome).rename({"key": "match_key"})
    print("header rows:", H.height, "distinct keys:", H["match_key"].n_unique())
    print("sample key:", H["match_key"][0])
    H.write_parquet(SCR / f"hdr_{year}.parquet")

    RE = scan_channel(tome, "round_end",
                      ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                       "win_reason_message", "player_count"])
    print("round_end rows:", RE.height, " distinct (match, round):",
          RE.select(["match_key", "round"]).unique().height)
    RE.write_parquet(SCR / f"re_{year}.parquet")

    FE = scan_channel(tome, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
                      where=pl.col("event_type") == "round_freeze_end")
    print("round_freeze_end rows:", FE.height, " distinct (match, round):",
          FE.select(["match_key", "round"]).unique().height)
    FE.write_parquet(SCR / f"fe_{year}.parquet")

    BS = scan_channel(tome, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"],
                      where=pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
    print("bomb rows by type:", BS.group_by("event_type").len().sort("event_type").to_dicts())
    BS.write_parquet(SCR / f"bomb_{year}.parquet")

print("\nwrote intermediates to", SCR)
