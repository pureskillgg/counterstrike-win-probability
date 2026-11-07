import sys, polars as pl, pyarrow.parquet as pq
BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
hdr = BASE + r"\header"
for name in ["header/dataframe_00000", "header/keyset_00000"]:
    p = BASE + "\\" + name.replace("/", "\\")
    md = pq.ParquetFile(p).metadata
    print(name, "rows", md.num_rows, "rowgroups", md.num_row_groups, "cols", md.num_columns)
    print(pq.ParquetFile(p).schema_arrow)
    print()
df = pl.read_parquet(BASE + r"\header\dataframe_00000")
print(df.shape)
print(df.head(3).to_dicts()[0] if df.height else None)
ks = pl.read_parquet(BASE + r"\header\keyset_00000")
print("keyset shape", ks.shape, ks.columns)
print(ks.head(3))
