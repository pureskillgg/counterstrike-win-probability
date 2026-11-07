import glob, os, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
H = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds\header")
files = sorted(glob.glob(str(H / "dataframe_*")))
print("header pages", len(files))
hd = pl.read_parquet(files[0])
print("header schema", hd.schema)
rf = sorted(glob.glob(str(TOME / "basic_round_end.2023-12-28,2024-02-25" / "dataframe_*")))
print("round_end pages", len(rf))
rd = pl.read_parquet(rf[0])
print("round_end schema", rd.schema)
t26 = sorted(glob.glob(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds\basic_round_end*"))
print("2026 round_end folders", t26[:2])
p1 = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\02"
p2 = r"E:\csds_final\csds\2026\10\csds_tail_recovered_20261011\csds\2024\02\02"
print("tail path exists", os.path.isdir(p1), "2026/10 path exists", os.path.isdir(p2))
if os.path.isdir(p1):
    k = sorted(os.listdir(p1))[0]
    print("sample uuid", k, os.listdir(os.path.join(p1, k)))
    rp = os.path.join(p1, k, "round_end")
    if os.path.exists(rp):
        r = pl.read_parquet(rp)
        print("raw round_end schema", r.schema, r.height)
