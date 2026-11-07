import json, polars as pl, collections
POS = "G:/pureskill/csds-cs2/data/combined/cache_pos_2026_trainctr.parquet"
p = pl.scan_parquet(POS)
print("pos_trainctr columns:", list(p.collect_schema().names()), flush=True)
MAN = "G:/pureskill/csds-cs2/data/combined/split_manifest.json"
man = json.load(open(MAN, encoding="utf-8"))["assign"]
c = collections.Counter()
for k, v in man.items():
    if k.startswith("csds/2026/"):
        c[v] += 1
print("manifest tags, keys with csds/2026/ prefix:", dict(c), flush=True)
print("manifest prefixes sample:", collections.Counter(k.split("/")[0] + "/" + k.split("/")[1] if k.count("/") >= 2 else k[:8] for k in man).most_common(6), flush=True)
