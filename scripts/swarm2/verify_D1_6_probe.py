"""verify D1-6 (swarm2 verifier): schema probe for player_death and bomb_state event types (2024), and
player_info rows per round by player_count (2024). Lazy reads with column projection only. No writes.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(20)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"


def f24(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


pd_f = f24("player_death")
print("player_death files:", len(pd_f))
print("player_death schema:", pl.read_parquet_schema(pd_f[0]))
bs = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "event_type"]) for f in f24("bomb_state")])
print("bomb_state event types:")
print(bs.group_by("event_type").len().sort("len", descending=True).collect())
pi = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_id_fixed", "team_code"]) for f in f24("player_info")])
RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_count"]) for f in f24("round_end")]).unique(
    ["match_key", "round"], keep="first")
pr = pi.group_by(["match_key", "round"]).agg(pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("nT"),
                                              (pl.col("team_code") == 3).sum().alias("nCT"))
j = RE.join(pr, on=["match_key", "round"], how="left")
print("player_info rows per round by player_count (2024):")
print(j.group_by("player_count").agg(pl.len().alias("rounds"), pl.col("np").median().alias("median_np"),
                                     pl.col("nT").median().alias("med_nT"), pl.col("nCT").median().alias("med_nCT"))
      .sort("rounds", descending=True).head(10))
