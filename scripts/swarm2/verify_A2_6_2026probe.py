# verify_A2_6 part 4 (read-only probe): is the 2026 '0 mismatches' comparator vacuous?
# Counts 2026 bomb_state event types and round-13 / bomb-outcome rows using the 2026 channel suffix.
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = r"G:\pureskill\csds-cs2"
T26 = os.path.join(BASE, "tomes", "2026-10-01_to_08", "tome", "csds")
dirs = sorted(d for d in os.listdir(T26) if d.startswith("basic_round_end.") or d.startswith("basic_bomb_state."))
print("2026 dirs:", dirs)


def read(prefix, cols):
    ds = [d for d in os.listdir(T26) if d.startswith(prefix + ".")]
    fs = []
    for d in ds:
        fs += sorted(glob.glob(os.path.join(T26, d, "dataframe_*")))
    print(prefix, "files:", len(fs), "dirs:", ds)
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed") if fs else None


RE = read("basic_round_end", ["match_key", "round", "winner_team_code", "player_count"])
BS = read("basic_bomb_state", ["match_key", "round", "event_type", "player_team_code"])
print("round_end rows:", RE.height, " player_count values:", RE["player_count"].unique().sort().to_list()[:10])
print("bomb_state rows:", BS.height)
print("bomb_state event_type counts:", BS.group_by("event_type").len().sort("len", descending=True).rows()[:30])
outcome = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).select(["match_key", "round"]).unique()
print("rounds with bomb_defused/bomb_exploded outcome (2026):", outcome.height)
print("round_end rows with round==13:", RE.filter(pl.col("round") == 13).height)
# 2026 mismatch count, without the 2024 player_count==20 filter (2026 player_count values are 2..11)
ro = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
planter = (BS.filter(pl.col("event_type") == "bomb_planted")
           .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_side")))
bo = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
j26 = (ro.join(bo, on=["match_key", "round"], how="inner")
       .join(planter, on=["match_key", "round"], how="left")
       .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
       .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
print("2026 outcome rounds joined to round_end:", j26.height, " mismatches:", int(j26["mismatch"].sum()))
print("2026 round_end player_count full set:", RE["player_count"].unique().sort().to_list(),
      " rows with player_count==20:", RE.filter(pl.col("player_count") == 20).height)
print("2026 winner_team_code values:", j26["winner_team_code"].unique().sort().to_list(),
      " planter_side values:", j26["planter_side"].unique().sort().to_list())
print("DONE")
