"""verify_D9_10 step 1: value probes for winner evidence fields (read-only, projected columns, stdout only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def scan(name, cols):
    fd = BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(sorted(glob.glob(str(fd / "dataframe_*"))))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in cols if c in have]).collect()


RE = scan("round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message", "legacy_code", "player_count"])
print("== round_end winner_team_code x win_reason_message")
print(RE.group_by(["winner_team_code", "win_reason_code", "win_reason_message"]).len().sort("len", descending=True))
print("== round_end legacy_code value counts")
print(RE.group_by("legacy_code").len().sort("len", descending=True).head(10))
print("== round_end player_count value counts")
print(RE.group_by("player_count").len().sort("len", descending=True).head(10))
RS = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase", "is_warmup"])
print("== round_state event_type value counts")
print(RS.group_by("event_type").len().sort("len", descending=True).head(20))
print("== round_state phase value counts")
print(RS.group_by("phase").len().sort("len", descending=True).head(20))
print("== round_state sample, one match first 12 rows")
mk = RS["match_key"][0]
print(RS.filter(pl.col("match_key") == mk).sort("tick").head(12))
