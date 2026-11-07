import sys, os
import pyarrow.parquet as pq
# argv: local tail file, total size, label
tail_path, total, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tail = open(tail_path, "rb").read()
sparse = tail_path + ".sparse"
with open(sparse, "wb") as f:
    f.seek(total - len(tail))
    f.write(tail)
md = pq.read_metadata(sparse)
print(f"== {label}: rows={md.num_rows} row_groups={md.num_row_groups} cols={md.num_columns}")
sch = md.schema.to_arrow_schema()
for fld in sch:
    print(f"   {fld.name} : {fld.type}")
os.remove(sparse)
