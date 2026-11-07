import glob, os, collections
import pyarrow.parquet as pq
ROOT = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
CH = ["item_equip", "item_pickup", "item_dropped", "item_refund", "rank_update"]
for ch in CH:
    pages = sorted(glob.glob(os.path.join(ROOT, f"basic_{ch}.2026-09-18,2026-10-08", "dataframe_*")))
    maps = []
    for p in pages:
        s = pq.ParquetFile(p).schema_arrow
        maps.append({f.name: str(f.type) for f in s})
    allc = []
    for m in maps:
        for k in m:
            if k not in allc: allc.append(k)
    print(f"== {ch} ({len(pages)} pages, {len(allc)} columns in union)")
    for c in allc:
        types = [m.get(c, "MISSING") for m in maps]
        if len(set(types)) > 1 or "MISSING" in types:
            print(f"  DRIFT {c}: " + ", ".join(f"p{i}={t}" for i, t in enumerate(types)))
    # order check
    orders = set(tuple(m.keys()) for m in maps)
    print("  distinct column orders:", len(orders))
    rows = [pq.ParquetFile(p).metadata.num_rows for p in pages]
    print("  rows per page:", rows, "sum", sum(rows))
