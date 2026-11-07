"""A3 probe: schema and sample values for the dedupe audit (read-only, 2026 tome)."""
import glob
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"


def files(channel):
    folder = TOME / ("header" if channel == "header" else f"basic_{channel}{SUFFIX}")
    return sorted(glob.glob(str(folder / "dataframe_*")))


def read(channel, cols=None):
    fs = files(channel)
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


hdr = read("header")
print("header schema:", hdr.schema)
print("header rows:", hdr.height, "unique key:", hdr["key"].n_unique())
print(hdr.head(3))

for ch in ["round_end", "player_death", "player_info"]:
    df = read(ch)
    print(f"\n{ch} schema: {df.schema}")
    print(f"{ch} rows: {df.height}")
    print(df.head(3))

de = read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code", "attacker_id"])
print("\nplayer_id dtype:", de.schema["player_id"], "null count:", de["player_id"].null_count())
print("distinct player_id:", de["player_id"].n_unique(), "sample:", de["player_id"].head(8).to_list())
print("distinct (match,player_id) per match mean:",
      de.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n")).select(pl.col("n").mean()).item())

keys = read("round_end", ["match_key"])["match_key"].unique().to_list()
print("\nsample match_key:", keys[:2])
ks = sorted(glob.glob(str(TOME / f"basic_round_end{SUFFIX}" / "keyset_*")))
print("keyset files round_end:", ks)
print("keyset sample:", pl.concat([pl.read_parquet(f) for f in ks])["_"].head(2).to_list())
