import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1)
ROOTS = {
    "2024": Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"),
    "2026": Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"),
}
def files(base, ch):
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))
for era, base in ROOTS.items():
    rs = pl.scan_parquet(files(base, "round_state")).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase"]).collect()
    print(f"==== {era} round_state rows={rs.height}")
    print(rs.group_by("event_type").len().sort("len", descending=True))
    print("score raw == score share:", (rs["t_score"] == rs["t_score_raw"]).mean(), (rs["ct_score"] == rs["ct_score_raw"]).mean())
    # one match sample
    m0 = rs["match_key"][0]
    print("sample match", m0)
    print(rs.filter(pl.col("match_key") == m0).filter(pl.col("event_type").str.contains("round_(freeze_end|end|start|official)|official|end")).sort("tick").head(60))
    if era == "2026":
        su = pl.scan_parquet(files(base, "score_update")).select(["match_key", "round", "tick", "team_code", "old_score", "new_score"]).collect()
        print("score_update team_code counts:", su.group_by("team_code").len().sort("team_code").rows())
        print(su.filter(pl.col("match_key") == m0).sort("tick").head(30))
