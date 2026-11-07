"""verify D6-6, step 00: schema and parquet writer metadata for the 2024 tome (footers only)."""
import glob
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
for ch in ["header", "round_end", "round_state", "bomb_state", "player_info"]:
    if ch == "header":
        fd = BASE / "header"
    else:
        fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    print(f"== {ch}: {len(files)} files")
    md = pq.ParquetFile(files[0]).metadata
    print("  created_by:", md.created_by, "| rows(file0):", md.num_rows, "| row_groups:", md.num_row_groups)
    print("  schema:", [f"{f.name}:{f.type}" for f in pq.ParquetFile(files[0]).schema_arrow])
    kv = md.metadata or {}
    print("  kv keys:", [k.decode(errors="replace") for k in kv.keys()][:12])
    cb = set()
    for f in files:
        cb.add(pq.ParquetFile(f).metadata.created_by)
    print("  distinct created_by across files:", cb)
