import sys, glob, os
import pyarrow.parquet as pq
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "dataframe_*")))
tot = 0
for f in files:
    md = pq.ParquetFile(f).metadata
    tot += md.num_rows
    print("FILE", os.path.basename(f), "rows", md.num_rows, "rowgroups", md.num_row_groups)
print("TOTAL_ROWS", tot)
sch = pq.ParquetFile(files[0]).schema_arrow
for fld in sch:
    print("  col:", fld.name, "|", str(fld.type))
