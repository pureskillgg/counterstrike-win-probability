"""D4 schema probe: columns and sample rows for the channels the tick-based round test needs.

Reads only the 2024 tome (G:) and the 2026 tome (G:). Prints schemas, event_type values for bomb and round
channels, and a few sample rows. No writes outside logs\\swarm2 (stdout is redirected there by the caller).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D4_schema_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}


def folder(base, suf, ch):
    if ch == "header":
        return base / "header"
    cands = sorted(p for p in base.glob(f"basic_{ch}.*"))
    if not cands:
        return None
    return cands[0]


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


for era, (base, suf) in TOMES.items():
    print(f"\n######## {era} ({base}) ########")
    chans = sorted(p.name for p in base.glob("basic_*"))
    print("channels:", [c.replace("basic_", "").split(".")[0] for c in chans])
    for ch in ["bomb_state", "bomb_defuse", "bomb_action", "round_state", "round_end", "round_start"]:
        fd = folder(base, suf, ch)
        if fd is None:
            print(f"\n[{era}] {ch}: MISSING")
            continue
        fs = files(fd)
        schema = pl.read_parquet_schema(fs[0])
        print(f"\n[{era}] {ch}: pages={len(fs)} schema={dict(schema)}")
        df = pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed")
        print(f"[{era}] {ch}: rows={df.height} matches={df['match_key'].n_unique()}")
        if "event_type" in df.columns:
            print(df.group_by("event_type").len().sort("len", descending=True))
        print(df.head(6))
        del df
