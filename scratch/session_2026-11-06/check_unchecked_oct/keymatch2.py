import json, collections, polars as pl, sys
G="G:/pureskill/csds-cs2/data/combined"
m=json.load(open(G+"/split_manifest.json"))["assign"]
def day(k):
    p=k.split("/"); return "-".join(p[1:4])
split_of = m
caches={"base":G+"/cache_snapshot_2026.parquet","ext":G+"/cache_snapshot_2026_ext.parquet","exp":G+"/cache_snapshot_2026_oct23_31.parquet"}
ser_split = pl.DataFrame({"match_key":list(m.keys()),"split":list(m.values()),"kday":[day(k) for k in m]})
for name,p in caches.items():
    g = (pl.scan_parquet(p).group_by("match_key").agg(pl.len().alias("rows"), pl.col("match_date").first().alias("mdate")).collect())
    j = g.join(ser_split, on="match_key", how="left").with_columns(pl.col("split").fill_null("none"))
    print("=====", name, "matches", g.height, "rows", g["rows"].sum())
    print(" by split (matches, rows):")
    for s in ["train","val","test","none"]:
        sub=j.filter(pl.col("split")==s)
        print("  ", s, sub.height, sub["rows"].sum())
    o = j.filter((pl.col("kday")>="2026-10-21")&(pl.col("kday")<="2026-10-25"))
    print(" Oct21-25 (by key path) in cache: matches", o.height, "rows", o["rows"].sum())
    for s in ["train","val","test"]:
        sub=o.filter(pl.col("split")==s)
        print("   ", s, sub.height, sub["rows"].sum())
    o21=j.filter(pl.col("kday")=="2026-10-21")
    print(" Oct21 by path in cache:", o21.height, "train", o21.filter(pl.col("split")=="train").height)
    # Holdout keys in cache by date
    nd = j.filter(pl.col("split")=="none")
    print(" unlabelled cache matches by key-day counts:", nd.group_by("kday").len().sort("kday").head(40).rows().__repr__()[:600] if nd.height else 0)
    print(" max kday in cache", j["kday"].max(), "min kday", j["kday"].min())
    mism = j.filter(pl.col("kday")!=pl.col("mdate").str.slice(0,10).str.replace_all("/","-"))
    print(" key-path day vs match_date mismatch count", mism.height)
