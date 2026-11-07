import json, collections, polars as pl
G="G:/pureskill/csds-cs2/data/combined"
m=json.load(open(G+"/split_manifest.json"))["assign"]
def day(k):
    p=k.split("/"); return "-".join(p[1:4])
base=set(pl.scan_parquet(G+"/cache_snapshot_2026.parquet").select("match_key").unique().collect()["match_key"].to_list())
b_tr={k for k in base if m.get(k)=="train"}; b_va={k for k in base if m.get(k)=="val"}; b_te={k for k in base if m.get(k)=="test"}
print("base cache by manifest split", len(b_tr), len(b_va), len(b_te), "unlabelled", len(base-set(m)))
print("manifest test total by year-month:", collections.Counter(day(k)[:7] for k,v in m.items() if v=="test").most_common(8))
ft=set(json.load(open("C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/data/benchmark/frozen_test_keys.json")))
print("frozen == base test split:", ft==b_te, len(ft), len(b_te), "diff", len(ft^b_te))
cf=set(json.load(open(G+"/centre_fit_train_keys.json")))
print("centre_fit == base train split:", cf==b_tr, len(cf), len(b_tr), "diff", len(cf^b_tr), "centre_fit in base?", len(cf & base))
vk=pl.read_parquet("C:/Users/<user>/sprint/matrix_trainctr/val_keys.parquet")
vset=set(vk["match_key"].unique().to_list())
print("matrix val keys == base val split:", vset==b_va, len(vset), len(b_va), "diff", len(vset^b_va))
print("matrix val keys dates min/max", min(day(k) for k in vset), max(day(k) for k in vset), "Oct21-25 n", sum(1 for k in vset if "2026-10-21"<=day(k)<="2026-10-25"))
print("matrix val rows", vk.height)
