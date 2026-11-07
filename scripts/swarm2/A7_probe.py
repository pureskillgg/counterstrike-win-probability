import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
T26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
def files(folder):
    return sorted(glob.glob(str(folder / "dataframe_*")))
for label, base, suf in (("2026", T26, ".2026-09-18,2026-10-08"), ("2024", T24, ".2023-12-28,2024-02-25")):
    for ch in ("round_state", "round_end", "score_update"):
        fd = base / f"basic_{ch}{suf}"
        if not fd.exists():
            print(label, ch, "MISSING"); continue
        fs = files(fd)
        print(label, ch, "files", len(fs), "schema", dict(pl.read_parquet_schema(fs[0])))
    hf = sorted(glob.glob(str(base / "header" / "dataframe_*")))
    print(label, "header files", len(hf), dict(pl.read_parquet_schema(hf[0])))
    hk = sorted(glob.glob(str(base / "header" / "*")))
    print(label, "header dir", [Path(x).name for x in hk][:5])
