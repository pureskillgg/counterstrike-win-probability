"""D7-5 probe: (a) header score conventions for 24-round matches (sanity for D7-4b);
(b) round_state ledger (t_score/ct_score at round_freeze_end) schema and per-round coverage.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
R = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_all.parquet"))
HDR = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "t_starters_score_final", "ct_starters_score_final"]).rename({"key": "match_key"}).with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))
mr = R.group_by("match_key").agg(pl.col("round").max().alias("maxr"), pl.col("round").n_unique().alias("nr"))
reg = mr.filter(pl.col("maxr") == 24).select("match_key")
W = (R.filter(pl.col("player_count") == 20).join(reg, on="match_key").select(["match_key", "round", "winner_team_code"]))
agg = W.group_by("match_key").agg(
    (pl.col("winner_team_code") == 2).sum().alias("w2"),
    (pl.col("winner_team_code") == 3).sum().alias("w3"),
    pl.col("round").n_unique().alias("nr"),
)
A = HDR.join(agg, on="match_key", how="inner")
print("24-round matches with header:", A.height)
print("rounds per match == 24:", (A["nr"] == 24).sum())
print("header t+ct == nr:", ((A["t_starters_score_final"] + A["ct_starters_score_final"]) == A["nr"]).sum())
print("header {t,ct} == {w2,w3} (as a set):", (((A["t_starters_score_final"] == A["w2"]) & (A["ct_starters_score_final"] == A["w3"])) |
      ((A["t_starters_score_final"] == A["w3"]) & (A["ct_starters_score_final"] == A["w2"]))).sum())
print("header t_starters == w2 (T code total):", (A["t_starters_score_final"] == A["w2"]).sum(),
      "| header t_starters == w3:", (A["t_starters_score_final"] == A["w3"]).sum())
print("sample header vs winner totals:", A.select(["t_starters_score_final", "ct_starters_score_final", "w2", "w3", "nr"]).head(8).rows())

print("\n== round_state channel ==")
RS_types = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_state{SUF}")]).select(["event_type"]).group_by("event_type").len().collect()
print("round_state event types:", RS_types.sort("event_type").rows())
RSs = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_state{SUF}")])
print("round_state schema:", RSs.collect_schema().names())
