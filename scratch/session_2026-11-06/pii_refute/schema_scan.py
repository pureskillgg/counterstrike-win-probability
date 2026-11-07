import sys, os, glob
import pyarrow.parquet as pq
roots = ["G:/pureskill", "G:/MeridianData"]
files = []
for r in roots:
    for dp, dn, fn in os.walk(r):
        if "envs" in dp.replace("\\", "/").split("/") or ".pixi" in dp:
            continue
        for f in fn:
            if f.endswith(".parquet"):
                files.append(os.path.join(dp, f))
files.sort()
print("TOTAL_PARQUET", len(files))
for f in files:
    try:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        cols = [(pf.schema_arrow.field(i).name, str(pf.schema_arrow.field(i).type)) for i in range(len(pf.schema_arrow.names))]
        short = f.replace("\\", "/").replace("G:/pureskill/", "")
        print(f"FILE {short} rows={md.num_rows} ncols={len(cols)}")
        print("  COLS " + ", ".join(f"{n}:{t}" for n, t in cols))
    except Exception as e:
        print("ERR", f, type(e).__name__)
