import polars as pl, sys, csv
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/"
CK = S + "check_unchecked_oct/"
G = "G:/pureskill/csds-cs2/"
batch = set(r["match_key"] for r in csv.DictReader(open(CK + "batch_v4_per_match.csv", encoding="utf-8")))
hb = set(l.strip() for l in open(G + "data/holdout_b/holdout_keys.txt", encoding="utf-8") if l.strip())
vk = pl.read_parquet("C:/Users/<user>/sprint/matrix_trainctr/val_keys.parquet")
print("cols", vk.columns, "dtypes", vk.dtypes, "rows", vk.height, flush=True)
print("head", vk.head(2).to_dicts(), flush=True)
if "match_key" in vk.columns:
    mk = set(vk["match_key"].to_list())
    print("distinct match_key", len(mk), "batch overlap", len(mk & batch), "HB overlap", len(mk & hb), flush=True)
    print("val rows per day (first 4 date tokens)", vk.with_columns(pl.col("match_key").str.slice(11, 10).alias("d")).group_by("d").n_unique().head(3) if False else "", flush=True)
