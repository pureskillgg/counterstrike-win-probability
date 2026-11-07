# verify_A2_4_pickraw.py: pick three build-9842 day-01 match keys to probe in the raw 2024 tail copy (read-only).
import os, json
import polars as pl
pl.Config.set_tbl_rows(20)
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
hdr = pl.scan_parquet(os.path.join(TOME, "header", "dataframe_00000")).select(["key", "match_id", "build_num", "match_date"]).collect()
x = hdr.filter((pl.col("build_num") == 9842) & (pl.col("key").str.contains("/2024/02/01/"))).head(3)
print(x.rows(), flush=True)
