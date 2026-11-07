# Probe: do the bomb outcome ticks in round_state agree with bomb_state, and do they carry the -429 offset vs round_end?
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
def scan(ch): return pl.scan_parquet(files(BASE / f"basic_{ch}{SUF}"))
re_ = scan("round_end").select(["match_key", "round", "tick", "second"]).collect()
print("round_end tick/second:", (re_.filter(pl.col("second") > 0).select((pl.col("tick") / pl.col("second")).median()).item()))
bs = scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(["match_key", "round", "tick", "second", "event_type"]).collect()
rs = scan("round_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(["match_key", "round", "tick", "second", "event_type"]).collect()
print("bomb_state exploded/defused:", bs.height, "| round_state exploded/defused:", rs.height)
print("bomb_state tick/second median:", bs.select((pl.col("tick") / pl.col("second")).median()).item(),
      "| round_state tick/second median:", rs.select((pl.col("tick") / pl.col("second")).median()).item())
print("bomb_state (match,round,event) dup count:", bs.height - bs.unique(["match_key", "round", "event_type"]).height,
      "| round_state dup count:", rs.height - rs.unique(["match_key", "round", "event_type"]).height)
# one row per (match, round): min tick of exploded/defused, as in the claim
bo = bs.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("bs_tick"), pl.col("event_type").first().alias("bs_ev"))
ro = rs.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("rs_tick"), pl.col("event_type").first().alias("rs_ev"))
J = (re_.select(["match_key", "round", "tick"]).rename({"tick": "end_tick"})
       .join(bo, on=["match_key", "round"], how="inner")
       .join(ro, on=["match_key", "round"], how="left"))
print("rounds with bomb_state outcome:", J.height, "| with round_state outcome:", J.filter(pl.col("rs_tick").is_not_null()).height)
J = J.with_columns((pl.col("bs_tick") - pl.col("end_tick")).alias("lag_bs"), (pl.col("rs_tick") - pl.col("end_tick")).alias("lag_rs"),
                   (pl.col("rs_tick") - pl.col("bs_tick")).alias("rs_minus_bs"))
print("lag_bs (bomb_state - round_end) top values:", J.group_by("lag_bs").len().sort("len", descending=True).head(6).rows())
print("lag_rs (round_state - round_end) top values:", J.filter(pl.col("lag_rs").is_not_null()).group_by("lag_rs").len().sort("len", descending=True).head(6).rows())
print("rs_minus_bs top values:", J.filter(pl.col("rs_minus_bs").is_not_null()).group_by("rs_minus_bs").len().sort("len", descending=True).head(8).rows())
print("event-type agreement (bs_ev == rs_ev) among matched:", J.filter(pl.col("rs_ev").is_not_null()).select((pl.col("bs_ev") == pl.col("rs_ev")).sum()).item(), "of", J.filter(pl.col("rs_ev").is_not_null()).height)
print("lag_rs == -429 count:", J.filter(pl.col("lag_rs") == -429).height, "| lag_bs == -429 count:", J.filter(pl.col("lag_bs") == -429).height)
print("round_state (match,round) with >1 bomb-outcome row:", rs.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height)
