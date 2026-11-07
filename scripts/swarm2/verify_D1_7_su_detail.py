"""verify D1-7 (swarm2 verifier): raw 2026 score_update events in the rounds that carry two +1 events (recomputed from raw)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(50)
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
RE = pl.scan_parquet(sorted(glob.glob(str(T26 / "basic_round_end*" / "dataframe_*")))).select(["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"]).collect()
su = pl.scan_parquet(sorted(glob.glob(str(T26 / "basic_score_update*" / "dataframe_*")))).collect()
su = su.with_columns((pl.col("new_score") - pl.col("old_score")).alias("delta"))
p1 = su.filter(pl.col("delta") == 1).group_by(["match_key", "round"]).agg(pl.len().alias("n_plus1"), pl.col("team_code").sort().alias("plus1_teams"))
J = RE.join(p1, on=["match_key", "round"], how="left")
two = J.filter(pl.col("n_plus1") == 2)
print("two-+1 rounds by round number:", two.group_by("round").len().sort("round").to_dicts())
print("two-+1 rounds: player_count distribution:", two.group_by("player_count").len().sort("len", descending=True).head(5).to_dicts())
print("two-+1 rounds: plus1 team pair distribution:", two.group_by(pl.col("plus1_teams").cast(pl.List(pl.Utf8))).len().to_dicts())
print("two-+1 rounds: winner in pair:", two.with_columns(pl.col("plus1_teams").list.contains(pl.col("winner_team_code")).alias("w")).group_by("w").len().to_dicts())
ex = two.filter(pl.col("round") == 27).select("match_key").head(3)["match_key"].to_list()
for mk in ex:
    print("\n==== 2026", mk)
    ev = su.filter((pl.col("match_key") == mk) & pl.col("round").is_between(26, 28)).sort(["round", "tick"])
    print(ev.select(["round", "tick", "team_code", "old_score", "new_score", "delta"]))
    print("winner by round:", J.filter((pl.col("match_key") == mk) & pl.col("round").is_between(26, 28)).select(["round", "winner_team_code", "plus1_teams", "n_plus1"]).sort("round").to_dicts())
mx = RE.group_by("match_key").agg(pl.col("round").max().alias("max_round"))
two_m = two.join(mx, on="match_key", how="left")
print("\nmax round of matches containing a two-+1 round:", two_m.group_by("max_round").len().sort("max_round").to_dicts())
print("\ncount of matches with max_round>=27 (all):", mx.filter(pl.col("max_round") >= 27).height, " with max_round>=33:", mx.filter(pl.col("max_round") >= 33).height)
