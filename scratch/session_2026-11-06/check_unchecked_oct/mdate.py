import json, collections, polars as pl
G="G:/pureskill/csds-cs2/data/combined"
m=json.load(open(G+"/split_manifest.json"))["assign"]
def pday(k):
    p=k.split("/"); return "-".join(p[1:4])
g=(pl.scan_parquet(G+"/cache_snapshot_2026_oct23_31.parquet").group_by("match_key")
   .agg(pl.col("match_date").first().alias("mdate"), pl.len().alias("rows")).collect())
g=g.with_columns(pl.col("mdate").str.slice(0,10).str.replace_all("/","-").alias("md"))
rows=g.to_dicts()
ct=collections.Counter()
ex=collections.defaultdict(list)
for r in rows:
    k=r["match_key"]; pd=pday(k); split=m.get(k,"none")
    if pd!=r["md"]:
        grp = "none" if split=="none" else ("manifest-Oct21-25" if "2026-10-21"<=pd<="2026-10-25" else "manifest-other")
        ct[(grp, pd[:10], r["md"][:10])]+=1
print("expanded cache keys where match_date day != key-path day:", sum(ct.values()))
for (grp,pd,md),n in sorted(ct.items(), key=lambda x:-x[1])[:25]:
    print(f"  {grp:20s} path={pd} match_date={md} n={n}")
# Count by match_date window
md_oct2125=[r for r in rows if "2026-10-21"<=r["md"]<="2026-10-25"]
print("by match_date Oct21-25: total", len(md_oct2125), "of which manifest-split:", collections.Counter(m.get(r["match_key"],"none") for r in md_oct2125))
print("  of which key-path Oct26-31 (holdout H):", sum(1 for r in md_oct2125 if pday(r["match_key"])>="2026-10-26"))
md_base=[r for r in rows if r["md"]<"2026-10-14"]
print("by match_date < Oct14 total", len(md_base))
