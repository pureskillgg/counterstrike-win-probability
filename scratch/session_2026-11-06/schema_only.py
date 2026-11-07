import sys
import pyarrow.parquet as pq

# Schema-level inspection only: column names, types, row counts.
# No cell values are read or printed.
for path in sys.argv[1:]:
    try:
        pf = pq.ParquetFile(path)
        sch = pf.schema_arrow
        print("FILE:", path.split("/")[-1], "| parent:", path.split("/")[-2][:8], "| rows:", pf.metadata.num_rows, "| row_groups:", pf.metadata.num_row_groups)
        for field in sch:
            print("   ", field.name, "::", str(field.type))
    except Exception as e:
        print("ERR", path.split("/")[-1], type(e).__name__)
