"""D2 probe 2: round_state event types, team_change coverage (2026), and score field behaviour at round 13."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds", "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def read(label, ch, cols=None):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    lf = pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*")))
    if cols:
        lf = lf.select(cols)
    return lf.collect()


for label in ["2024", "2026"]:
    rs = read(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase", "is_warmup"])
    print(f"\n== {label} round_state event types:")
    print(rs.group_by("event_type").len().sort("len", descending=True).rows())
    print("phase values:", rs.group_by("phase").len().sort("len", descending=True).rows()[:10])
    fe = rs.filter(pl.col("event_type") == "round_freeze_end")
    print("freeze_end rows", fe.height, "distinct (match,round)", fe.select(["match_key", "round"]).unique().height)
    sample = fe.filter(pl.col("match_key") == fe["match_key"][0]).sort("round").select(["round", "tick", "t_score", "ct_score"])
    print("sample match freeze_end scores:")
    print(sample.head(30))
    if label == "2026":
        tc = read(label, "team_change", ["match_key", "round", "player_id", "old_team_code", "new_team_code", "is_bot"])
        print("team_change rows", tc.height, "by round (first 30):")
        print(tc.group_by("round").len().sort("round").head(30).rows())
        print("team_change old->new:", tc.group_by(["old_team_code", "new_team_code"]).len().rows())
