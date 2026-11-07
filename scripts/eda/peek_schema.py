import sys, glob, os
import pyarrow.parquet as pq

ROOT = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
CHANNELS = ["item_equip", "item_pickup", "item_dropped", "item_refund", "rank_update"]

for ch in CHANNELS:
    d = os.path.join(ROOT, f"basic_{ch}.2026-09-18,2026-10-08")
    pages = sorted(glob.glob(os.path.join(d, "dataframe_*")))
    print(f"== {ch}: {len(pages)} pages")
    for p in pages:
        f = pq.ParquetFile(p)
        md = f.metadata
        print(f"  {os.path.basename(p)} rows={md.num_rows} rowgroups={md.num_row_groups}")
    schema = pq.ParquetFile(pages[0]).schema_arrow
    print("  columns:", len(schema.names))
    for fld in schema:
        print(f"    {fld.name}: {fld.type}")
    hdr = os.path.join(d, "header")
    print("  header folder:", os.listdir(hdr) if os.path.isdir(hdr) else None)
