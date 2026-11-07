import glob, os
import pyarrow.parquet as pq

ROOT = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
d = os.path.join(ROOT, "basic_item_equip.2026-09-18,2026-10-08")
ks = sorted(glob.glob(os.path.join(d, "keyset_*")))
print("keyset files:", len(ks), os.path.basename(ks[0]))
t = pq.read_table(ks[0])
print("keyset schema:", t.schema)
print("keyset rows:", t.num_rows)
print(t.slice(0, 5).to_pydict())
for hdr in ["G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds/header"]:
    for f in sorted(glob.glob(os.path.join(hdr, "*"))):
        print("header:", os.path.basename(f), os.path.getsize(f))
dh = os.path.join(d, "header")
for f in sorted(glob.glob(os.path.join(dh, "*"))):
    print("chan header:", os.path.basename(f), os.path.getsize(f))
hp = os.path.join(dh, "dataframe_00000")
if os.path.exists(hp):
    ht = pq.read_table(hp)
    print("chan header df schema names:", ht.schema.names[:50], "rows", ht.num_rows)
