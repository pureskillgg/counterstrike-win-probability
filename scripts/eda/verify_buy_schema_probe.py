import sys
import pyarrow.parquet as pq
import pyarrow as pa

paths = [
    r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet",
    r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet",
    r"G:\pureskill\csds-cs2\data\state_sample\round_buy_types.parquet",
    r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds\header\dataframe_00000",
    r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds\header\keyset_00000",
]
for p in paths:
    print("==", p)
    try:
        f = pq.ParquetFile(p)
        print("rows:", f.metadata.num_rows, "row_groups:", f.num_row_groups)
        print(f.schema_arrow)
    except Exception as e:
        print("ERR", type(e).__name__, str(e)[:300])
        try:
            with open(p, "rb") as fh:
                head = fh.read(200)
            print("head bytes:", head[:200])
        except Exception as e2:
            print("ERR2", e2)
    print()
