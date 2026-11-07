import glob, os, sys
import pyarrow.parquet as pq
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
def show(folder):
    for kind in ("dataframe", "keyset"):
        for f in sorted(glob.glob(os.path.join(folder, kind + "_*"))):
            pf = pq.ParquetFile(f)
            md = pf.metadata
            print(os.path.basename(f), "rows", md.num_rows, "rowgroups", md.num_row_groups, "cols", md.num_columns, "bytes", os.path.getsize(f))
            print("  schema:", [(pf.schema_arrow.field(i).name, str(pf.schema_arrow.field(i).type)) for i in range(len(pf.schema_arrow.names))][:60])
            kv = md.metadata or {}
            for k, v in kv.items():
                if k.startswith(b"pandas") or len(v) > 300:
                    print("  meta", k, v[:300])
                else:
                    print("  meta", k, v)
print("== header"); show(os.path.join(T, "header"))
print("== round_end"); show(os.path.join(T, "basic_round_end.2026-09-18,2026-10-08"))
