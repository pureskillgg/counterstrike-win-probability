import sys, glob
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
pi = pl.read_parquet(sorted(glob.glob(str(sorted(BASE.glob("basic_player_info.*"))[0] / "dataframe_*")))[0],
                     columns=["match_key","round","player_id","player_id_fixed","team_code"])
print("rows", pi.height, "matches", pi["match_key"].n_unique())
print("null team", pi["team_code"].null_count(), "null pid_fixed", pi["player_id_fixed"].null_count(), "null pid", pi["player_id"].null_count())
print("team_code values", pi["team_code"].value_counts().sort("team_code"))
per_round = pi.group_by(["match_key","round"]).agg(pl.len().alias("n"), pl.col("team_code").eq(2).sum().alias("t"), pl.col("team_code").eq(3).sum().alias("ct"))
print("players per round dist:", per_round["n"].value_counts().sort("n"))
print("rounds per match dist (top):", pi.group_by("match_key").agg(pl.col("round").n_unique().alias("r"))["r"].value_counts().sort("count", descending=True).head(15))
print("round min/max:", pi["round"].min(), pi["round"].max())
# example match
m = pi["match_key"].unique().sort()[0]
ex = pi.filter(pl.col("match_key")==m).sort(["round","player_id"])
print(m)
print(ex.pivot(on="round", index="player_id", values="team_code", aggregate_function="first").sort("player_id"))
