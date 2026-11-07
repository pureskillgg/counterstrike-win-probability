"""D3 probe: round-by-round round_end winner vs round_state freeze_end scores for two matches."""
import glob
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
RE = g_read("round_end", ["match_key", "round", "tick", "winner_team_code"])
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
for key in ["csds/2024/02/02/548405a3-c739-4b1e-8265-67c818037c41/csds",
            "csds/2024/02/17/3389858a-38d6-4df3-bbac-a77d8f43bbb5/csds"]:
    re_m = RE.filter(pl.col("match_key") == key).sort("round")
    rs_m = RS.filter((pl.col("match_key") == key) & (pl.col("event_type") == "round_freeze_end")).sort("tick")
    print("\nMATCH", key)
    print(re_m.join(rs_m.select(["round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).unique("round", keep="first"),
                    on="round", how="left", suffix="_fe").select(
        ["round", "winner_team_code", "tick", "tick_fe", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).rows())
    evs = RS.filter(pl.col("match_key") == key)["event_type"].value_counts().sort("count", descending=True)
    print("round_state event types:", evs.rows()[:12])
