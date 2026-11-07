import sys, pyarrow as pa, pyarrow.parquet as pq
for path in sys.argv[1:]:
    buf = open(path, 'rb').read()
    md = pq.read_metadata(pa.BufferReader(buf))
    sch = pq.read_schema(pa.BufferReader(buf))
    print(f"== {path.split(chr(92))[-1]}: rows={md.num_rows} row_groups={md.num_row_groups} cols={len(sch.names)}")
    for n, t in zip(sch.names, sch.types):
        print(f"   {n} : {t}")
