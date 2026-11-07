import polars as pl, sys, csv
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/"
CK = S + "check_unchecked_oct/"
G = "G:/pureskill/csds-cs2/data/combined/"
batch = set(r["match_key"] for r in csv.DictReader(open(CK + "batch_v4_per_match.csv", encoding="utf-8")))
kl = pl.DataFrame({"match_key": list(batch)}).lazy()
for f in ["cache_snapshot_2026_oct23_31.parquet", "cache_snapshot_2026_ext.parquet"]:
    lf = pl.scan_parquet(G + f).select("match_key")
    u = lf.select(pl.col("match_key").n_unique().alias("nu"), pl.col("match_key").null_count().alias("nulls"), pl.len().alias("rows")).collect()
    print(f, u.rows(), flush=True)
    kn = lf.select(pl.col("match_key").unique()).collect().height
    print("  distinct (unique) keys", kn, flush=True)
tc = pl.scan_parquet(G + "cache_pos_2026_trainctr.parquet").select(["match_key", "round"]).join(kl, on="match_key", how="semi").select(pl.len().alias("rows"), pl.col("match_key").n_unique().alias("keys")).collect()
print("trainctr batch rows/keys", tc.rows(), flush=True)
