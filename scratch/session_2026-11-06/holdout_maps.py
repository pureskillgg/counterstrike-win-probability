import sys
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
M = pl.read_parquet(r"G:\pureskill\csds-cs2\data\combined\matches.parquet")
print("columns:", [c for c in M.columns if c in ("match_key","upload_day","in_scope","dup_keeper","map_name","match_date","era","max_rounds_final")])
cols = [c for c in ("upload_day","in_scope","dup_keeper","map_name") if c in M.columns]
if "upload_day" in M.columns:
    h = M.filter(pl.col("upload_day").cast(pl.Utf8).str.contains(r"2026[/-]10[/-](2[6-9]|3[01])"))
    print("matches uploaded Oct 26-31:", h.height)
    if "in_scope" in M.columns and "dup_keeper" in M.columns:
        k = h.filter(pl.col("in_scope") & pl.col("dup_keeper"))
        print("in scope and keeper:", k.height)
        print(k.group_by("map_name").len().sort("len", descending=True))
