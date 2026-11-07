import glob
import os

import polars as pl
import pyarrow.parquet as pq

TOME = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
STATE = "G:/pureskill/csds-cs2/data/state_sample"


def pages(channel):
    folder = os.path.join(TOME, f"basic_{channel}.COMBINED")
    return sorted(glob.glob(os.path.join(folder, "dataframe_*")))


for ch in ["player_hurt", "player_death", "round_end"]:
    ps = pages(ch)
    print("==", ch, "pages:", len(ps))
    f = pq.ParquetFile(ps[0])
    print("rows in page0:", f.metadata.num_rows, "row groups:", f.metadata.num_row_groups)
    print(f.schema_arrow)
    print()

for name in ["players_1hz.parquet", "rounds.parquet", "round_buy_types.parquet", "place_names.parquet"]:
    p = os.path.join(STATE, name)
    f = pq.ParquetFile(p)
    print("==", name, "rows:", f.metadata.num_rows)
    print(f.schema_arrow)
    print()
