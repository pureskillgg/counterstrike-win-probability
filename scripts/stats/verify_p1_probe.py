"""Schema probe for independent verification of P1 (read-only)."""
import glob
import os

import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
SAMPLE = r"G:\pureskill\csds-cs2\data\state_sample"


def chan_files(prefix):
    pats = glob.glob(os.path.join(TOME, f"basic_{prefix}.*", "dataframe_*"))
    return sorted(pats)


for ch in ["round_end", "round_state", "player_death", "weapon_fire", "player_hurt", "player_info", "item_equip"]:
    fs = chan_files(ch)
    print(ch, "files:", len(fs))
    if fs:
        lf = pl.scan_parquet(fs[0])
        print("  schema:", lf.collect_schema())
        print("  first-file rows:", lf.select(pl.len()).collect().item())

for f in ["players_1hz.parquet", "rounds.parquet", "round_buy_types.parquet"]:
    p = os.path.join(SAMPLE, f)
    lf = pl.scan_parquet(p)
    print(f, "schema:", dict(lf.collect_schema()))
    print(f, "rows:", lf.select(pl.len()).collect().item())
