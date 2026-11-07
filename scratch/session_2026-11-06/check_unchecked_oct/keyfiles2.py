import json, collections, polars as pl, sys, os
G="G:/pureskill/csds-cs2/data/combined"
m=json.load(open(G+"/split_manifest.json"))["assign"]
def day(k):
    p=k.split("/"); return "-".join(p[1:4])
base_tr={k for k,v in m.items() if v=="train"}
print("manifest train total", len(base_tr))
def load_json_keys(p):
    x=json.load(open(p))
    if isinstance(x,dict):
        for kk in ["keys","match_keys","train_keys","test_keys"]:
            if kk in x: return set(x[kk]), list(x.keys())
        return set(x.keys()), list(x.keys())
    return set(x), None
ft,ftk=load_json_keys("C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/data/benchmark/frozen_test_keys.json")
print("frozen keys", len(ft), "dict keys", ftk)
test_base={k for k,v in m.items() if v=="test" and day(k)<="2026-10-13"}
print("frozen == manifest base test?", ft==test_base, "frozen dates", min(day(k) for k in ft), max(day(k) for k in ft), "n in Oct21-25", sum(1 for k in ft if "2026-10-21"<=day(k)<="2026-10-25"))
cf,cfk=load_json_keys(G+"/centre_fit_train_keys.json")
print("centre_fit keys", len(cf), "dict keys", cfk, "== manifest train (base)?", cf==base_tr, "n Oct21-25", sum(1 for k in cf if "2026-10-21"<=day(k)<="2026-10-25"), "max day", max(day(k) for k in cf))
vk=pl.read_parquet("C:/Users/<user>/sprint/matrix_trainctr/val_keys.parquet")
print("val_keys cols", vk.columns, vk.height)
col=vk.columns[0]
vset=set(vk[col].to_list())
val_base={k for k,v in m.items() if v=="val" and day(k)<="2026-10-13"}
print("matrix val keys == manifest base val?", vset==val_base, len(vset), "n Oct21-25", sum(1 for k in vset if "2026-10-21"<=day(k)<="2026-10-25"), "dates", min(day(k) for k in vset), max(day(k) for k in vset))
hb=[l.strip() for l in open(G+"/../holdout_b/holdout_keys.txt") if l.strip()]
hbs=set(hb)
test_o=({k for k,v in m.items() if v=="test" and "2026-10-21"<=day(k)<="2026-10-25"})
print("holdout_b (G) count", len(hb), len(hbs), "== manifest Oct21-25 test?", hbs==test_o, "dates", min(day(k) for k in hbs), max(day(k) for k in hbs))
hb2=[l.strip() for l in open("C:/Users/<user>/sprint/holdout_oct26_31_keys.parquet") ] if False else None
hh=pl.read_parquet("C:/Users/<user>/sprint/holdout_oct26_31_keys.parquet")
print("holdout H parquet cols", hh.columns, hh.height)
hcol=[c for c in hh.columns if "key" in c.lower()] or hh.columns[:1]
hset=set(hh[hcol[0]].to_list())
print("holdout H keys", len(hset), "dates min/max", min(day(k) for k in hset), max(day(k) for k in hset), "Oct21-25 count", sum(1 for k in hset if "2026-10-21"<=day(k)<="2026-10-25"))
print("overlap holdout H with manifest", len(hset & set(m.keys())))
print("overlap holdout B with base train/val/test", len(hbs & set(k for k in m if day(k)<="2026-10-13")))
