# verify_A2_4_rawprobe.py: read-only probe of one raw 2024 tail-copy match (build 9842 in the tome) whose channels are files.
import os
import polars as pl
pl.Config.set_tbl_rows(30)
RAW = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\01\m_bb0ad43bd38ae7c458c92d68"
for ch in ["round_state", "round_start", "round_end", "tick"]:
    p = os.path.join(RAW, ch)
    print(ch, "exists:", os.path.exists(p), "size:", os.path.getsize(p) if os.path.exists(p) else None, flush=True)
    if not os.path.exists(p):
        continue
    with open(p, "rb") as fh:
        head = fh.read(8)
    print("   first bytes:", head, flush=True)
    try:
        df = pl.read_parquet(p)
        print("   parquet rows", df.height, "cols", df.columns[:14], flush=True)
        if "round" in df.columns:
            print("   round labels:", df.group_by("round").len().sort("round").head(10).rows(), flush=True)
        if "event_type" in df.columns:
            r1 = df.filter(pl.col("round") == 1)
            print("   round==1 rows:", r1.height, r1.group_by("event_type").len().rows()[:12], flush=True)
    except Exception as e:
        print("   not parquet:", type(e).__name__, str(e)[:160], flush=True)
print("DONE", flush=True)
