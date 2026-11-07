import glob, os
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
CH = ["item_dropped", "item_pickup", "round_end", "player_death", "weapon_fire", "round_state", "round_start"]

for ch in CH:
    folder = os.path.join(BASE, f"basic_{ch}.COMBINED")
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    print("=====", ch, "pages:", len(files))
    if not files:
        continue
    lf = pl.scan_parquet(files)
    schema = lf.collect_schema()
    print("schema:", dict(schema))
    n = lf.select(pl.len()).collect().item()
    print("rows:", n)
    keys = lf.select(pl.col("match_key").str.slice(0, 9).alias("era")).group_by("era").agg(pl.len().alias("n")).collect()
    print("era counts:", keys.to_dicts())
    head = lf.head(3).collect()
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(head)
