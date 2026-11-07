import json, collections, polars as pl, sys
G="G:/pureskill/csds-cs2/data/combined"
m=json.load(open(G+"/split_manifest.json"))
a=m["assign"]
def day(k):
    p=k.split("/")
    return "-".join(p[1:4]) if len(p)>4 else None
c=collections.Counter()
for k,v in a.items():
    d=day(k)
    c[(v, d)] +=1
tot=collections.Counter(v for v in a.values())
print("manifest totals", tot, len(a))
octs=collections.Counter()
for (v,d),n in c.items():
    if d and d.startswith("2026-10") or (d and d.startswith("2026-09")):
        pass
for lab in ["train","val","test"]:
    n=sum(n for (v,d),n in c.items() if v==lab and d and "2026-10-21"<=d<="2026-10-25")
    print("Oct21-25", lab, n)
n=sum(n for (v,d),n in c.items() if d and "2026-10-21"<=d<="2026-10-25")
print("Oct21-25 total keys", n)
print("Oct14-20 by split", {lab: sum(n for (v,d),n in c.items() if v==lab and d and "2026-10-14"<=d<="2026-10-20") for lab in ["train","val","test"]})
print("Oct26-31 keys by split", {lab: sum(n for (v,d),n in c.items() if v==lab and d and "2026-10-26"<=d<="2026-10-31") for lab in ["train","val","test"]})
print("max key date in manifest", max(day(k) for k in a))
print("Oct21 train/val/test", {lab: sum(n for (v,d),n in c.items() if v==lab and d=="2026-10-21") for lab in ["train","val","test"]})
