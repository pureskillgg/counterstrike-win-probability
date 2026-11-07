"""Independent verifier for MA: print schemas and small samples of the three channels used."""
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
PATHS = {
    "header": BASE + r"\header\dataframe_00000",
    "round_end": BASE + r"\basic_round_end.2026-09-18,2026-10-08\dataframe_00000",
    "round_state": BASE + r"\basic_round_state.2026-09-18,2026-10-08\dataframe_00000",
}

for name, path in PATHS.items():
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    print(f"== {name}: {len(schema)} columns")
    for col, dtype in schema.items():
        print(f"   {col}: {dtype}")
    n = lf.select(pl.len()).collect().item()
    print(f"   rows: {n}")
    print(lf.head(3).collect())
    print()

rs = pl.scan_parquet(PATHS["round_state"])
if "event_type" in rs.collect_schema().names():
    print(rs.group_by("event_type").agg(pl.len().alias("n")).sort("n", descending=True).collect())
