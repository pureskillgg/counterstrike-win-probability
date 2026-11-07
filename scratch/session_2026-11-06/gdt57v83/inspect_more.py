import json, polars as pl
HO2 = "C:/Users/<user>/sprint/holdout_oct26_31/keys.parquet"
CACHE = "G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026.parquet"
MAN = "G:/pureskill/csds-cs2/data/combined/split_manifest.json"
s = pl.scan_parquet(HO2)
print("holdout_oct26_31/keys.parquet schema:", dict(s.collect_schema()), "rows(footer):", s.select(pl.len()).collect().item(), flush=True)
c = pl.scan_parquet(CACHE)
print("cache_snapshot_2026 columns:", list(c.collect_schema().names())[:12], "... total", len(c.collect_schema().names()), flush=True)
print("has match_date:", "match_date" in c.collect_schema().names(), "has map_name:", "map_name" in c.collect_schema().names(), flush=True)
m = json.load(open(MAN, encoding="utf-8"))
print("split_manifest top keys:", list(m.keys())[:15], flush=True)
for k, v in m.items():
    if isinstance(v, dict):
        print(" ", k, "dict len", len(v), "sample value types", type(next(iter(v.values()))).__name__ if v else None, flush=True)
    elif isinstance(v, list):
        print(" ", k, "list len", len(v), flush=True)
    else:
        print(" ", k, type(v).__name__, str(v)[:120], flush=True)
print("HB description:", json.load(open("G:/pureskill/csds-cs2/data/combined/holdout_22_25_test_keys.json", encoding="utf-8"))["description"], flush=True)
