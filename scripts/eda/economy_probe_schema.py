# Probe: schema and footer metadata only (no row reads) for the header tome,
# one raw player_status file on E:, and the round_state / round_end tome pages.
import sys, time, glob, os
import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
hdr = os.path.join(TOME, "header", "dataframe_00000")
print("header schema:")
f = pq.ParquetFile(hdr)
print(f.schema_arrow)
print("header rows", f.metadata.num_rows, "row groups", f.metadata.num_row_groups)

for ch in ["round_end", "round_state"]:
    d = glob.glob(os.path.join(TOME, "basic_%s.*" % ch))[0]
    pages = sorted(glob.glob(os.path.join(d, "dataframe_*")))
    print(ch, d, "pages", len(pages))
    for p in pages[:1]:
        g = pq.ParquetFile(p)
        print(g.schema_arrow.names)
        print("rows", g.metadata.num_rows, "row groups", g.metadata.num_row_groups)

E_MATCH = r"E:\csds_final\csds\2026\10\01"
t0 = time.time()
items = sorted(os.listdir(E_MATCH))[:1]
print("E day 01 first ids", items, "list time", round(time.time() - t0, 3))
mid = os.path.join(E_MATCH, items[0])
print("contents", os.listdir(mid))
for ch in os.listdir(mid):
    p = os.path.join(mid, ch)
    print(ch, os.path.getsize(p))
ps = os.path.join(mid, "player_status")
t0 = time.time()
g = pq.ParquetFile(ps)
print("player_status rows", g.metadata.num_rows, "row groups", g.metadata.num_row_groups, "footer time", round(time.time() - t0, 3))
print(g.schema_arrow)
