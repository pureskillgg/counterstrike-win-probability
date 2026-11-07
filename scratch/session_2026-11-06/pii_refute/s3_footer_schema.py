import sys, io, glob
import pyarrow.parquet as pq
OUT = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/pii_refute"
for p in sorted(glob.glob(OUT + "/tail_cs2data_combined_*.parquet")):
    data = open(p, "rb").read()
    try:
        md = pq.read_metadata(io.BytesIO(data))
        sch = md.schema.to_arrow_schema()
        names = [(sch.field(i).name, str(sch.field(i).type)) for i in range(len(sch.names))]
        print("S3", p.split("tail_")[-1], "rows", md.num_rows, "ncols", len(names))
        print("  COLS " + ", ".join(f"{n}:{t}" for n, t in names))
    except Exception as e:
        print("ERR", p.split("tail_")[-1], type(e).__name__, str(e)[:200])
