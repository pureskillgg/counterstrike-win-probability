"""D3 probe: schema, match_key format and header of the 2024 G: tome, plus one raw E: match.

Read-only. Reads G: tome parquet pages and one E: parquet file per channel.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D3_probe_schema.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(120)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
E_ROOT = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")


def pages(ch):
    if ch == "header":
        folder = TOME / "header"
    else:
        folder = TOME / f"basic_{ch}{SUF}"
    return sorted(glob.glob(str(folder / "dataframe_*")))


def schema(ch):
    fs = pages(ch)
    print(f"[{ch}] pages={len(fs)}")
    for f in fs:
        print("  ", Path(f).name, pl.read_parquet_schema(f))
    return fs


print("== G: tome channels available (2024) ==")
for ch in ["header", "round_end", "round_state", "bomb_state", "bomb_action", "bomb_defuse", "round_start",
           "player_info", "player_death"]:
    print(ch, "pages:", len(pages(ch)))

print("\n== schemas ==")
for ch in ["header", "round_end", "round_state", "bomb_state", "bomb_action", "bomb_defuse"]:
    schema(ch)

print("\n== header sample ==")
H = pl.concat([pl.read_parquet(f) for f in pages("header")], how="vertical_relaxed")
print("header rows:", H.height)
print(H.head(3))

print("\n== round_end sample ==")
RE = pl.concat([pl.read_parquet(f) for f in pages("round_end")], how="vertical_relaxed")
print("round_end rows:", RE.height)
print(RE.head(5))
print("sample match_key:", RE["match_key"][0])

print("\n== bomb_state sample ==")
BS = pl.concat([pl.read_parquet(f) for f in pages("bomb_state")], how="vertical_relaxed")
print("bomb_state rows:", BS.height)
print(BS.head(5))
print(BS["event_type"].value_counts().sort("count", descending=True))

print("\n== bomb_action sample ==")
BA = pl.concat([pl.read_parquet(f) for f in pages("bomb_action")], how="vertical_relaxed")
print("bomb_action rows:", BA.height)
print(BA.head(5))

print("\n== round_state sample ==")
RS = pl.concat([pl.read_parquet(f) for f in pages("round_state")], how="vertical_relaxed")
print("round_state rows:", RS.height)
print(RS.head(5))
print(RS["event_type"].value_counts().sort("count", descending=True).head(40))

print("\n== raw E: one match, round_end and bomb_state schemas ==")
m = sorted(glob.glob(str(E_ROOT / "12" / "*")))[0]
print("E match dir:", m)
for ch in ["round_end", "bomb_state", "bomb_action", "round_state", "header"]:
    p = Path(m) / ch
    if p.exists():
        print(f"[{ch}] {pl.read_parquet_schema(str(p))}")
    else:
        print(f"[{ch}] MISSING")
