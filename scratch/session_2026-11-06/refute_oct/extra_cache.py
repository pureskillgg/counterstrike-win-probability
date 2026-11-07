import polars as pl, sys, json
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
base = "G:/pureskill/csds-cs2/data/combined/"
extra = set(pl.read_csv(S + "s3_extra_oct21_25_keys.csv")["match_key"].to_list())
res = {}
for f in ["cache_snapshot_2026_oct23_31.parquet", "cache_snapshot_2026_ext.parquet", "cache_snapshot_2026.parquet", "cache_pos_2026.parquet", "cache_pos_2026_trainctr.parquet", "cache_pos_2026_before_oct14_31.parquet"]:
    ks = set(pl.scan_parquet(base + f).select("match_key").unique().collect()["match_key"].to_list())
    res[f] = len(extra & ks)
    print(f, "extra156 present:", len(extra & ks), flush=True)
json.dump(res, open(S.replace("check_unchecked_oct/", "refute_oct/") + "refute_extra_cache.json", "w"), indent=1)
