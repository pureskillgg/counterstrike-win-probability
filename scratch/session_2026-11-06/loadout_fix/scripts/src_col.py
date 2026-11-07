import json, collections
import pyarrow.parquet as pq
F = "/home/sagemaker-user/adam/builder/final"
keys = json.load(open(F + "/export_clean_v2/match_keys.json"))
ks = set(keys)
cm = pq.read_table(F + "/clean_matches.parquet", columns=["match_key", "src", "in_scope", "month", "gkeeper", "dup_keeper"]).to_pylist()
print("src counts all", collections.Counter(r["src"] for r in cm))
exp = [r for r in cm if r["match_key"] in ks]
print("src counts in export", collections.Counter(r["src"] for r in exp))
print("export rows in clean", len(exp), "distinct", len({r["match_key"] for r in exp}))
bk = [r["match_key"] for r in exp if r["src"] == "bucket"]
print("export & src=bucket", len(bk), "distinct", len(set(bk)))
print("export src=bucket by month", sorted(collections.Counter(r["month"] for r in exp if r["src"] == "bucket").items()))
print("in_scope among export", collections.Counter(str(r["in_scope"]) for r in exp))
print("in_scope among export&bucket", collections.Counter(str(r["in_scope"]) for r in exp if r["src"] == "bucket"))
