"""verify D1-7 (swarm2 verifier): event-type and score-field probe."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"


def scan(base, name, suf=""):
    files = sorted(glob.glob(str(base / f"basic_{name}{suf}*" / "dataframe_*")))
    return pl.scan_parquet(files)


for lab, base, suf in [("2024", T24, SUF24), ("2026", T26, "")]:
    bs = scan(base, "bomb_state", suf).group_by("event_type").agg(pl.len().alias("n")).sort("n", descending=True).collect()
    print(f"\n{lab} bomb_state event_type:\n", bs)
    rs = scan(base, "round_state", suf).group_by("event_type").agg(pl.len().alias("n")).sort("n", descending=True).head(10).collect()
    print(f"{lab} round_state event_type (top 10):\n", rs)

# score_update (2026 only): distribution of team_code and delta
su = scan(T26, "score_update").with_columns((pl.col("new_score") - pl.col("old_score")).alias("delta")).group_by(["team_code", "delta"]).agg(pl.len().alias("n")).sort(["team_code", "delta"]).collect()
print("\n2026 score_update team_code x delta:\n", su)

# t_score vs t_score_raw difference in 2024 round_state
d = (scan(T24, "round_state", SUF24).select(["match_key", "round", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
     .with_columns(((pl.col("t_score") != pl.col("t_score_raw")) | (pl.col("ct_score") != pl.col("ct_score_raw"))).alias("diff"))
     .group_by(["round"]).agg(pl.len().alias("n"), pl.col("diff").sum().alias("n_diff")).sort("round").collect())
print("\n2024 round_state t_score vs t_score_raw differences by round (rounds 20..34):\n", d.filter(pl.col("round").is_between(20, 34)))
d26 = (scan(T26, "round_state").select(["match_key", "round", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
       .with_columns(((pl.col("t_score") != pl.col("t_score_raw")) | (pl.col("ct_score") != pl.col("ct_score_raw"))).alias("diff"))
       .group_by(["round"]).agg(pl.len().alias("n"), pl.col("diff").sum().alias("n_diff")).sort("round").collect())
print("\n2026 round_state t_score vs t_score_raw differences by round (rounds 20..34):\n", d26.filter(pl.col("round").is_between(20, 34)))
