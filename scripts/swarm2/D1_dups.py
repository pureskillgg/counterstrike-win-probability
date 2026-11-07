import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(60)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
f = sorted(glob.glob(str(BASE / f"basic_player_info{SUF}" / "dataframe_*")))[0]
PI = pl.read_parquet(f, columns=["match_key","round","player_id_fixed","team_code"])
k = ["match_key","round","player_id_fixed"]
d = PI.filter(pl.len().over(k) > 1)
print("dup rows:", d.height, " dup keys:", d.select(k).unique().height)
print("dup keys with conflicting team_code:", d.group_by(k).agg(pl.col("team_code").n_unique().alias("nt")).filter(pl.col("nt")>1).height)
print("dup group sizes:", d.group_by(k).len()["len"].value_counts().to_dicts())
print("dup rows per match (top 15):")
print(d.group_by("match_key").len().sort("len", descending=True).head(15))
print("distinct matches with dups:", d["match_key"].n_unique())
print("dup rows by round (top):")
print(d.group_by("round").len().sort("len", descending=True).head(10))
RE = pl.read_parquet(sorted(glob.glob(str(BASE / f"basic_round_end{SUF}" / "dataframe_*")))[0], columns=["match_key","round","winner_team_code","player_count"])
dr = RE.filter(pl.len().over(["match_key","round"]) > 1)
print("round_end dup rows:"); print(dr)
