"""Schema peek for the weapons channel group (metadata only, no row reads)."""
import glob
import os

import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CHANNELS = ["weapon_fire", "weapon_action", "grenade_state", "grenade_bounce",
            "molotov_state", "molotov_fire"]

for ch in CHANNELS:
    folders = glob.glob(os.path.join(TOME, f"basic_{ch}.*"))
    pages = []
    for f in folders:
        pages += sorted(glob.glob(os.path.join(f, "dataframe_*.parquet")) +
                        glob.glob(os.path.join(f, "dataframe_*")))
    pages = sorted(set(pages))
    print(f"== {ch}: folders={len(folders)} pages={len(pages)}")
    if not pages:
        continue
    total = 0
    for p in pages:
        total += pq.ParquetFile(p).metadata.num_rows
    print(f"   total rows (metadata): {total}")
    schema = pq.ParquetFile(pages[0]).schema_arrow
    for field in schema:
        print(f"   {field.name}: {field.type}")
