import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
T26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
def scan(folder):
    return pl.scan_parquet(str(folder / "dataframe_*"))
for label, base, suf in (("2026", T26, ".2026-09-18,2026-10-08"), ("2024", T24, ".2023-12-28,2024-02-25")):
    rs = scan(base / f"basic_round_state{suf}")
    print("==", label)
    print(rs.group_by("event_type").agg(pl.len().alias("n")).sort("n", descending=True).collect())
    fe = rs.filter(pl.col("event_type") == "round_freeze_end").collect()
    print("freeze_end rows", fe.height, "unique (match,round)", fe.select(["match_key","round"]).unique().height)
    print("t_score null", fe["t_score"].null_count(), "ct_score null", fe["ct_score"].null_count())
    print(fe.select(["round","t_score","ct_score","t_score_raw","ct_score_raw","phase","is_warmup"]).head(12))
    # relation raw vs normalized
    print("t_score==t_score_raw", (fe["t_score"]==fe["t_score_raw"]).mean(), "ct equal", (fe["ct_score"]==fe["ct_score_raw"]).mean())
    print("phase values", fe.group_by("phase").len())
    print("warmup", fe.group_by("is_warmup").len())
    print("round dist sample", fe.group_by("round").len().sort("round").head(3))
