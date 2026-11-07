"""verify D7-5 probe: within-round score events in basic_round_state (no halftime swap inside a round).
Checks which event types carry t_score/ct_score and whether freeze_end(r) -> post-round event(r) gives a winner."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
RS = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_state{SUF}")]).select(
    ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase", "is_warmup"]).collect()
print("event types x (rounds with that type) and score-null counts:")
print(RS.group_by("event_type").agg(pl.len().alias("n"), pl.col("round").n_unique().alias("rounds"),
      pl.col("t_score").null_count().alias("t_null"), pl.col("match_key").n_unique().alias("matches")).sort("event_type"))
# per (match, round): which event types are present
fe = RS.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick", "t_score", "ct_score"]).unique(subset=["match_key", "round"], keep="first")
for et in ["round_officially_ended", "round_end", "cs_win_panel_round", "round_announce_last_round_half", "round_announce_match_point"]:
    ev = RS.filter(pl.col("event_type") == et).select(["match_key", "round", "tick", "t_score", "ct_score"])
    dup = ev.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
    j = fe.join(ev.unique(subset=["match_key", "round"], keep="first"), on=["match_key", "round"], how="inner", suffix="_ev")
    if j.height == 0:
        print(f"[{et}] no rows joined to freeze_end"); continue
    dt = (j["t_score_ev"] - j["t_score"]).to_list(); dc = (j["ct_score_ev"] - j["ct_score"]).to_list()
    from collections import Counter
    print(f"[{et}] events={ev.height} dup(match,round)={dup} joined={j.height} | (dt,dc) counter top:", Counter(zip(dt, dc)).most_common(6))
    print(f"[{et}] tick - freeze_end tick: median", j.select((pl.col("tick_ev") - pl.col("tick") if "tick_ev" in j.columns else pl.col("tick")).median()).item() if False else "see below")
