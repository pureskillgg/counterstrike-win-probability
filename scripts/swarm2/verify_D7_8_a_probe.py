"""verify_D7_8 step A: schema and coverage probe on the raw 2024 tome (no D7_* outputs read).
Writes nothing except stdout (captured into logs/swarm2/verify_D7_8.log by the shell call).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


for ch in ["round_state", "round_end", "bomb_state"]:
    fd = BASE / f"basic_{ch}{SUF}"
    fs = files(fd)
    print(f"== {ch}: {len(fs)} files")
    if fs:
        print("schema:", dict(pl.read_parquet_schema(fs[0])))
        print("total bytes:", sum(Path(f).stat().st_size for f in fs))

# round_state event types and rows
RS_ev = (pl.concat([pl.scan_parquet(f, ) .select(["event_type"]) for f in files(BASE / f"basic_round_state{SUF}")])
         .group_by("event_type").len().collect())
print("\nround_state event_type counts:", RS_ev.sort("event_type").rows())

FE = (pl.concat([pl.scan_parquet(f).select(["match_key", "round", "t_score", "ct_score", "event_type"])
                 for f in files(BASE / f"basic_round_state{SUF}")])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
print("\nfreeze_end rows:", FE.height,
      "unique (match,round):", FE.select(["match_key", "round"]).unique().height,
      "unique match_keys:", FE["match_key"].n_unique())
dup = (FE.group_by(["match_key", "round"]).agg(pl.len().alias("n"),
                                              pl.col("t_score").n_unique().alias("nt"),
                                              pl.col("ct_score").n_unique().alias("nc"))
       .filter(pl.col("n") > 1))
print("duplicated (match,round) freeze_end keys:", dup.height,
      "| of which conflicting scores:", dup.filter((pl.col("nt") > 1) | (pl.col("nc") > 1)).height)
print("null t_score:", FE["t_score"].null_count(), "null ct_score:", FE["ct_score"].null_count())
print("round range:", FE["round"].min(), FE["round"].max())
print("round value counts (first 40):", FE.group_by("round").len().sort("round").head(40).rows())

RE = (pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"])
                 for f in files(BASE / f"basic_round_end{SUF}")]).collect())
print("\nround_end rows:", RE.height, "unique (match,round):", RE.select(["match_key", "round"]).unique().height)
print("winner_team_code counts:", RE.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("player_count counts (top):", RE.group_by("player_count").len().sort("len", descending=True).head(12).rows())
print("matches with round_end:", RE["match_key"].n_unique(),
      "| with freeze_end:", FE["match_key"].n_unique())
print("round_end (match,round) without freeze_end:",
      RE.select(["match_key", "round"]).join(FE.select(["match_key", "round"]), on=["match_key", "round"], how="anti").height)
print("freeze_end (match,round) without round_end:",
      FE.select(["match_key", "round"]).join(RE.select(["match_key", "round"]), on=["match_key", "round"], how="anti").height)

H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "map_name", "build_num",
                                             "t_starters_score_final", "ct_starters_score_final"])
               for f in files(BASE / "header")])
print("\nheader rows:", H.height, "unique key:", H["key"].n_unique())
print("build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
print("null final scores:", H["t_starters_score_final"].null_count(), H["ct_starters_score_final"].null_count())
