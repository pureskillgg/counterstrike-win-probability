import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(80)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def fl(ch): return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
for ch in ["player_info", "round_end", "bomb_state", "player_personal"]:
    for f in fl(ch):
        print(ch, Path(f).name, round(Path(f).stat().st_size/1e6, 1), "MB")
pi = pl.scan_parquet(fl("player_info")[0]).select(["match_key","round","player_id_fixed","team_code"])
print("player_info rows:", pi.select(pl.len()).collect().item())
s = pi.group_by("match_key").agg(pl.col("round").n_unique().alias("nr"), pl.col("player_id_fixed").n_unique().alias("np")).collect()
print("matches in player_info:", s.height)
print(s.describe())
m = pi.filter(pl.col("match_key").str.contains("2024/02/13/")).head(0).collect()
ex = pi.head(40).collect()
print(ex)
print("team_code values:", pi.select(pl.col("team_code").unique()).collect().to_series().sort().to_list())
