import json, collections, os
import pyarrow.parquet as pq
import pyarrow.compute as pc
F = "/home/sagemaker-user/adam/builder/final"
keys = json.load(open(F + "/export_clean_v2/match_keys.json"))
ks = set(keys)
bm = pq.read_table(F + "/bucket_matches.parquet")
lm = pq.read_table(F + "/laptop_matches.parquet")
cm = pq.read_table(F + "/clean_matches.parquet")
print("bucket cols", bm.column_names)
print("laptop cols", lm.column_names)
print("clean cols", cm.column_names)
B = set(bm.column("match_key").to_pylist())
L = set(lm.column("match_key").to_pylist())
C = set(cm.column("match_key").to_pylist())
def ym(k): return k[4:11]
sets = {
 "export": ks,
 "export & bucket": ks & B,
 "export & laptop": ks & L,
 "export & bucket - laptop": (ks & B) - L,
 "export & laptop - bucket": (ks & L) - B,
 "export & bucket & laptop": ks & B & L,
 "export - bucket - laptop": ks - B - L,
 "export - bucket": ks - B,
}
for n, s in sets.items():
    c = collections.Counter(ym(k) for k in s)
    print(f"{n:28s} {len(s):7d}  ", dict(sorted(c.items())))
# clean_matches flags
for col in ["in_scope", "channels_complete", "month"]:
    if col in cm.column_names:
        vals = collections.Counter(str(v) for v in cm.column(col).to_pylist()).most_common(8)
        print(col, vals)
