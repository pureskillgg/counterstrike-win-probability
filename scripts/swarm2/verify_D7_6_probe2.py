"""verify D7-6 probe2: which round_state event types carry scores; round_end coverage vs date; one post match detail."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
RSf = sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))
RS = pl.concat([pl.scan_parquet(f) for f in RSf]).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase", "is_warmup"]).collect()
print("round_state rows:", RS.height)
print("score non-null by event_type:")
print(RS.group_by("event_type").agg(pl.len().alias("n"), pl.col("t_score").is_not_null().sum().alias("t_nn")).sort("event_type").rows())
H = pl.read_parquet(str(BASE / "header")) if False else None
HF = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
print("header files:", len(HF))
HD = pl.concat([pl.scan_parquet(f).select(["key", "match_date", "map_name", "build_num", "server_name", "match_id"]) for f in HF]).collect()
print("header rows:", HD.height, "unique key:", HD["key"].n_unique())
HD = HD.with_columns(pl.col("match_date").str.slice(0, 19).str.to_datetime(format="%Y-%m-%dT%H:%M:%S", time_unit="us").alias("mdt"))
RE = RS.filter(pl.col("event_type") == "round_end").select(["match_key", "round"]).unique()
FE = RS.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round"]).unique()
mt = HD.select(["key", "mdt"]).rename({"key": "match_key"})
fe_m = FE.join(mt, on="match_key", how="left").with_columns(pl.col("mdt").dt.strftime("%m-%d").alias("d"))
re_m = RE.join(mt, on="match_key", how="left").with_columns(pl.col("mdt").dt.strftime("%m-%d").alias("d"))
a = fe_m.group_by("d").agg(pl.len().alias("freeze_end_rounds")).sort("d")
b = re_m.group_by("d").agg(pl.len().alias("round_end_rounds")).sort("d")
print(a.join(b, on="d", how="full", coalesce=True).sort("d").rows())
