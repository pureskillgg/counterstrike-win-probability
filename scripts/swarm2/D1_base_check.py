import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def fl(ch): return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
def rd(ch, cols): return pl.concat([pl.read_parquet(f, columns=cols) for f in fl(ch)], how="vertical_relaxed")
RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"])
BS = rd("bomb_state", ["match_key", "round", "event_type", "player_id_fixed", "player_team_code"])
PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"])
print("round_end rows:", RE.height, " dup (match,round) in round_end:", RE.height - RE.unique(["match_key","round"]).height)
print("player_info rows:", PI.height, " dup (match,round,pid):", PI.height - PI.unique(["match_key","round","player_id_fixed"]).height)
print("player_info null pid:", PI["player_id_fixed"].null_count(), " null team:", PI["team_code"].null_count())
print("bomb_state event types:", BS["event_type"].value_counts().sort("count", descending=True).to_dicts())
outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused","bomb_exploded"]))
           .group_by(["match_key","round"]).agg(pl.col("event_type").unique().alias("evs")))
print("outcome rows with both defused and exploded:", outcome.filter(pl.col("evs").list.len()>1).height)
j = (RE.filter(pl.col("player_count")==20).join(outcome, on=["match_key","round"], how="inner")
     .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
     .with_columns((pl.col("bomb_side")!=pl.col("winner_team_code")).alias("mismatch")))
print("rounds with outcome (player_count==20):", j.height, " mismatches:", int(j["mismatch"].sum()))
print("total round_end player_count values:", RE["player_count"].value_counts().sort("player_count").to_dicts())
