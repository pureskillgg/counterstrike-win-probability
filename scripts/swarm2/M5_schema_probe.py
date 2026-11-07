import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
TOME = Path("G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds")
SUF = ".2026-09-18,2026-10-08"
def files(name):
    return sorted(glob.glob(str(TOME / name / "dataframe_*")))
hf = files("header")
print("header pages", len(hf))
h = pl.scan_parquet(hf)
print(h.collect_schema())
print(h.head(3).collect())
rf = files("basic_round_end" + SUF)
print("round_end pages", len(rf))
r = pl.scan_parquet(rf)
print(r.collect_schema())
print(r.head(3).collect())
print("polars", pl.__version__)
