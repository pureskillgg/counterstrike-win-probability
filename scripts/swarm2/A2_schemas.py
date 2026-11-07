import os, pyarrow.parquet as pq
BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
tot = 0
for d in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, d, "dataframe_00000")
    if not os.path.isfile(p):
        print("NO DATAFRAME", d); continue
    pf = pq.ParquetFile(p)
    md = pf.metadata
    sz = os.path.getsize(p)
    tot += sz
    names = pf.schema_arrow.names
    print(f"{d}: rows={md.num_rows} rg={md.num_row_groups} cols={len(names)} bytes={sz} ")
    print("   ", names)
print("total bytes", tot)
