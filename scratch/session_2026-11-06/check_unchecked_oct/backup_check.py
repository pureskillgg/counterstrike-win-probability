import json, collections, polars as pl
G="G:/pureskill/csds-cs2/data/combined"
bk=json.load(open(G+"/_backup_pre_22_26_20261027/split_manifest.json"))
print("backup keys top", list(bk.keys())[:10])
a=bk.get("assign",bk)
def day(k):
    p=k.split("/"); return "-".join(p[1:4])
print("backup n", len(a), collections.Counter(a.values()))
print("backup Oct21-25 by split", collections.Counter(v for k,v in a.items() if "2026-10-21"<=day(k)<="2026-10-25"))
print("backup Oct21 train", sum(1 for k,v in a.items() if day(k)=="2026-10-21" and v=="train"), "Oct22-25 keys", sum(1 for k in a if "2026-10-22"<=day(k)<="2026-10-25"))
ex=set(json.load(open("/dev/null")) if False else [])
