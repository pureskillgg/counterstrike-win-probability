import polars as pl, sys
E = "E:/csds_final/csds/2026/09/28/m_partial/"
for f in ["player_info", "header", "player_status"]:
    try:
        df = pl.read_parquet(E + f, n_rows=3)
        print(f, "schema:", df.schema, flush=True)
        print(df.head(2), flush=True)
    except Exception as e:
        print(f, "ERR", type(e).__name__, str(e)[:300], flush=True)
