# verify_A1_3_workaround.py
# Tests whether a single lazy polars scan can read item_equip's 8 pages once the claimed workaround
# is applied: integer->float casting plus absorbing the missing def_index column (pages 2-7).
# Aggregates only; memory stays small. Expected values come from the native pyarrow counts computed in
# verify_A1_3_tome.py and verify_A1_3_naive2.py (player_id_fixed nulls = 1649; def_index populated = 1,428,397;
# total rows = 9,585,945; def_index rows where the column is absent = 6,996,880 = pages 2-7 rows).
import glob, os, inspect
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
pages = sorted(glob.glob(os.path.join(TOME, "basic_item_equip.2026-09-18,2026-10-08", "dataframe_*")))

print("polars", pl.__version__)
print("scan_parquet params:", [p for p in inspect.signature(pl.scan_parquet).parameters if "missing" in p or "cast" in p or "extra" in p])

variants = [
    ("cast allow-float + missing_columns=insert",
     dict(cast_options=pl.ScanCastOptions(integer_cast="allow-float"), missing_columns="insert")),
    ("missing_columns=insert only", dict(missing_columns="insert")),
]
for label, kw in variants:
    try:
        agg = pl.scan_parquet(pages, **kw).select(
            pl.len().alias("n"),
            pl.col("player_id_fixed").null_count().alias("id_nulls"),
            pl.col("player_id_fixed").cast(pl.Float64).max().alias("id_max"),
            pl.col("def_index").is_not_null().sum().alias("def_populated"),
            pl.col("def_index").null_count().alias("def_nulls")).collect()
        print(f"{label}: OK -> {agg.row(0, named=True)}")
    except Exception as e:
        print(f"{label}: FAIL {type(e).__name__}: {str(e).splitlines()[0][:200]}")

print("expected (native): n=9585945 id_nulls=1649 def_populated=1428397 def_nulls=8157548 (= 6698 + 1153970 + 6996880)")
print("DONE workaround")
