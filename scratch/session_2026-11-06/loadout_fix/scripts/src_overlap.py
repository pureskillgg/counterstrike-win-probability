import json, collections, os
import pyarrow.parquet as pq
F = "/home/sagemaker-user/adam/builder/final"
keys = json.load(open(F + "/export_clean_v2/match_keys.json"))
ks = set(keys)
for name in ["bucket_matches.parquet", "laptop_matches.parquet", "clean_matches.parquet", "plants_bucket.parquet"]:
    t = pq.read_table(os.path.join(F, name))
    cols = t.column_names
    print(name, "rows", t.num_rows, "cols", cols[:12])
    kc = [c for c in cols if "key" in c.lower()]
    if kc:
        col = kc[0]
        vals = set(t.column(col).to_pylist())
        print("  key col", col, "distinct", len(vals), "in export_v2", len(vals & ks))
with open(F + "/split_manifest_v2.json", encoding="utf-8") as fh:
    m = json.load(fh)
print("manifest top keys", list(m.keys())[:10])
