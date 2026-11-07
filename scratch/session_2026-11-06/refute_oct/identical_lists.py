import polars as pl, sys, json
sys.stdout.reconfigure(encoding="utf-8")
G = "G:/pureskill/csds-cs2/data/"
hb_lap = set(l.strip() for l in open(G + "holdout_b/holdout_keys.txt", encoding="utf-8") if l.strip())
hb_comb = json.load(open(G + "combined/holdout_22_25_test_keys.json", encoding="utf-8"))
print("laptop holdout_keys.txt", len(hb_lap), "combined json type", type(hb_comb).__name__, flush=True)
if isinstance(hb_comb, dict):
    print("  dict keys", list(hb_comb.keys())[:6], flush=True)
    vals = hb_comb.get("keys") or hb_comb.get("match_keys") or next((v for v in hb_comb.values() if isinstance(v, list)), [])
else:
    vals = hb_comb
print("  combined json keys", len(vals), "equal to laptop txt:", set(vals) == hb_lap, flush=True)
a = pl.read_parquet("C:/Users/13144/sprint/holdout_oct26_31_keys.parquet")
b = pl.read_parquet("C:/Users/13144/sprint/holdout_oct26_31/keys.parquet")
print("oct26-31 files", a.columns, a.height, "|", b.columns, b.height, flush=True)
print("  same key set:", set(a[a.columns[0]].to_list()) == set(b[b.columns[0]].to_list()), flush=True)
