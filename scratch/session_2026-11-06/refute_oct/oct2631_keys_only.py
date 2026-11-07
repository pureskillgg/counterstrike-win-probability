import polars as pl, sys
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/refute_oct/"
k = pl.read_parquet("C:/Users/13144/sprint/holdout_oct26_31_keys.parquet", columns=None)
print("cols", k.columns, "rows", k.height, flush=True)
col = [c for c in k.columns if "key" in c.lower()][0]
pl.DataFrame({"match_key": k[col].to_list()}).write_csv(S + "refute_oct2631_keys.csv")
print("written", flush=True)
