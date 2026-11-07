"""verify D5_10, step 0: schema and coverage probe for the 2024 tome (read-only).

Prints column names and dtypes for the tables this verifier reads, row counts for the event types
used, and the keyset coverage per channel. Writes nothing except stdout.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def keyset(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "keyset_*")))
    s = set()
    for f in fs:
        s |= set(pl.read_parquet(f).to_series(0).to_list())
    return s


for ch in ["header", "round_end", "round_state", "round_start", "bomb_state", "player_info"]:
    fs = sorted(glob.glob(str(BASE / ("header" if ch == "header" else f"basic_{ch}{SUF}") / "dataframe_*")))
    sch = pl.read_parquet_schema(fs[0])
    print(f"[{ch}] files={len(fs)} schema={dict(sch)}")

hdr = pl.concat([pl.scan_parquet(f).select(["key", "match_date", "map_name", "match_type", "tick_rate",
                                             "t_starters_score_final", "ct_starters_score_final"])
                 for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))]).collect()
print("header rows:", hdr.height, "unique keys:", hdr["key"].n_unique())
print("match_type dist:", hdr.group_by("match_type").len().sort("len", descending=True).head(8).to_dicts())
print("map_name dist (top):", hdr.group_by("map_name").len().sort("len", descending=True).head(5).to_dicts())
print("tick_rate dist:", hdr.group_by("tick_rate").len().to_dicts())

RS_ev = pl.concat([pl.scan_parquet(f).select(["event_type"]) for f in files("round_state")]).group_by(
    "event_type").len().collect().sort("len", descending=True)
print("round_state event types (rows):")
print(RS_ev)

RE_n = pl.concat([pl.scan_parquet(f).select(["match_key", "round"]) for f in files("round_end")]).collect()
print("round_end rows:", RE_n.height, "distinct match-round:", RE_n.unique().height)

for ch in ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state",
           "team_change", "player_disconnect", "bot_takeover"]:
    ks = keyset(ch)
    have = set(pl.concat([pl.scan_parquet(f).select(["match_key"]).unique() for f in files(ch)]).collect()
               .to_series().to_list())
    print(f"[{ch}] keyset keys {len(ks)}; matches with rows {len(have)}; rows-without-keyset {len(have - ks)}")
print("keyset(header-like key set) from round_end:", len(keyset("round_end")))
