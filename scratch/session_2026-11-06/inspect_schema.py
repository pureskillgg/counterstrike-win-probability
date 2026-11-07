import pyarrow.parquet as pq
import os

B = "/home/sagemaker-user/adam/builder"
paths = [
    f"{B}/final/pos_parts/bucket_c000.parquet",
    f"{B}/data/c000/matches.parquet",
    f"{B}/data/c000/rounds.parquet",
    f"{B}/final/export_clean_v2/match_keys.json",
]
for p in paths:
    if p.endswith(".parquet"):
        f = pq.ParquetFile(p)
        print("==", p, "rows=", f.metadata.num_rows, "groups=", f.metadata.num_row_groups)
        print(f.schema_arrow)
    else:
        print("==", p, os.path.getsize(p), "bytes")
