"""D3-5 verifier, step H: round-by-round trace of one complete match where winner totals fail the header."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
M = sys.argv[1] if len(sys.argv) > 1 else "csds/2024/02/12/m_85654a833cd91589e79a7ba1/csds"


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


H = rd("header", ["key", "t_starters_score_final", "ct_starters_score_final", "match_date", "map_name"]).filter(pl.col("key") == M)
print(H)
RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count", "win_reason_code", "tick"]).filter(pl.col("match_key") == M)
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"]).filter(pl.col("match_key") == M)
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by("round").agg(pl.col("t_score").first().alias("t"), pl.col("ct_score").first().alias("ct"),
                             pl.col("phase").first().alias("ph")))
BS = rd("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("match_key") == M)
bo = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by("round").agg(
    pl.col("event_type").unique().alias("bomb"))
t = (RE.sort("round").join(fe, on="round", how="left").join(bo, on="round", how="left")
     .select(["round", "winner_team_code", "player_count", "win_reason_code", "t", "ct", "ph", "bomb"]))
print(t)
print("round_end rows:", RE.height, "| player_count values:", RE["player_count"].value_counts().rows())
print("round_end duplicates per round:", RE.group_by("round").len().filter(pl.col("len") > 1).rows())
