"""M4: peek at existing place-name and 1 Hz sample schemas (footers and small heads only)."""
import polars as pl
for p in [r"G:\pureskill\csds-cs2\data\state_sample\place_names.parquet",
          r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet",
          r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"]:
    lf = pl.scan_parquet(p)
    print("==", p.split("\\")[-1], "rows:", lf.select(pl.len()).collect().item())
    print(lf.collect_schema())
    print(lf.head(3).collect())
    print()
