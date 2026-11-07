"""Footer-only schema and row-count check. Reads parquet metadata; never reads row values."""
import glob
import os

import pyarrow.parquet as pq

ROOT = r"G:\pureskill\csds-cs2\tomes\combined\tome\csds"
TABLES = [
    "basic_player_info.COMBINED",
    "basic_player_inputs.COMBINED",
    "basic_player_name.COMBINED",
    "basic_player_personal.COMBINED",
    "basic_player_connect.COMBINED",
    "basic_rank_update.COMBINED",
]

for t in TABLES:
    d = os.path.join(ROOT, t)
    pages = sorted(glob.glob(os.path.join(d, "dataframe_*")))
    if not pages:
        print(f"{t}: no dataframe pages")
        continue
    total_rows = 0
    for p in pages:
        total_rows += pq.ParquetFile(p).metadata.num_rows
    first = pq.ParquetFile(pages[0])
    print(f"{t}: pages={len(pages)} total_rows_from_footers={total_rows}")
    for field in first.schema_arrow:
        print(f"    {field.name}: {field.type}")
    print()
