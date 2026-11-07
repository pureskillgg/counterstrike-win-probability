# A1: what a naive multi-page read resolves for drifting channels (column set, dtype of *_id_fixed, null count of def_index).
import os, glob
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
for c, col in [("basic_item_equip", "def_index"), ("basic_item_pickup", "player_id_fixed"), ("basic_player_hurt", "attacker_id_fixed")]:
    pages = sorted(glob.glob(os.path.join(T, c + ".2026-09-18,2026-10-08", "dataframe_*")))
    try:
        lf = pl.scan_parquet(pages)
        df = lf.collect()
        print(c, "naive scan: rows", df.height, "cols", df.width, "dtype", col, df.schema.get(col), "nulls", df.get_column(col).null_count() if col in df.columns else "absent")
    except Exception as e:
        print(c, "naive scan FAILED:", type(e).__name__, str(e).splitlines()[0][:160])
    try:
        d2 = pl.read_parquet(pages, use_pyarrow=True) if False else pl.read_parquet(pages)
        print(c, "read_parquet(list): rows", d2.height, "cols", d2.width, "dtype", d2.schema.get(col), "nulls", d2.get_column(col).null_count() if col in d2.columns else "absent")
    except Exception as e:
        print(c, "read_parquet(list) FAILED:", type(e).__name__, str(e).splitlines()[0][:160])
