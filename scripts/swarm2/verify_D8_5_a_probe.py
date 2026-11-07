"""verify D8-5 (probe): round_end row conflicts across ALL player_count values, and header metadata.

Read-only. Reads 2024 tome header and round_end only (column projection). Prints to stdout.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_a_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def fo(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def rd(fd, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fo(fd)], how="vertical_relaxed")


def rch(ch, cols):
    return rd(BASE / f"basic_{ch}{SUF}", cols)


H = rd(BASE / "header", ["key", "map_name", "match_date", "match_type", "server_name", "providence",
                         "number_of_points", "tick_rate", "t_starters_score_final", "ct_starters_score_final"])
print(f"header rows {H.height}; unique key {H['key'].n_unique()}")
print("match_type counts:", H.group_by("match_type").len().sort("len", descending=True).rows())
print("providence counts:", H.group_by("providence").len().sort("len", descending=True).rows()[:6])
print("tick_rate counts:", H.group_by("tick_rate").len().rows())
print("server_name top:", H.group_by("server_name").len().sort("len", descending=True).head(8).rows())

RE = rch("round_end", ["match_key", "round", "winner_team_code", "player_count"])
print(f"\nround_end rows (all player_count) {RE.height}")
print("player_count counts:", RE.group_by("player_count").len().sort("len", descending=True).head(10).rows())
g = RE.group_by(["match_key", "round"]).agg(
    pl.len().alias("n"),
    pl.col("winner_team_code").n_unique().alias("nw"),
    pl.col("player_count").n_unique().alias("npc"),
)
print(f"(match,round) groups {g.height}; groups with >1 row {g.filter(pl.col('n') > 1).height}; "
      f"groups with conflicting winner {g.filter(pl.col('nw') > 1).height}")
multi = g.filter(pl.col("n") > 1)
print("multi-row group size counts:", multi.group_by("n").len().rows())
print("multi-row groups with mixed player_count:", multi.filter(pl.col("npc") > 1).height)
conf = g.filter(pl.col("nw") > 1)
if conf.height:
    print("conflicting groups sample:")
    print(conf.head(10))
# the winner codes present
print("winner_team_code values:", RE.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("match keys in round_end not in header:", RE.filter(~pl.col("match_key").is_in(H["key"].to_list()))["match_key"].n_unique())
print("\nDONE")
