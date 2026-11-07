import sys, os
import pyarrow.parquet as pq
d = sys.argv[1]
for ch in ["player_info", "player_name", "player_personal", "player_death", "csds", "header"]:
    p = os.path.join(d, ch)
    if os.path.isdir(p):
        fs = [os.path.join(p, f) for f in os.listdir(p) if f.endswith(".parquet")]
        if not fs:
            print(ch, "dir with no parquet:", os.listdir(p)[:3]); continue
        f = fs[0]
    elif os.path.isfile(p):
        f = p
    else:
        print(ch, "missing"); continue
    pf = pq.ParquetFile(f)
    sch = pf.schema_arrow
    cols = [f"{sch.field(i).name}:{sch.field(i).type}" for i in range(len(sch.names))]
    print(f"CHANNEL {ch} rows={pf.metadata.num_rows} ncols={len(cols)}")
    print("   COLS " + ", ".join(cols))
