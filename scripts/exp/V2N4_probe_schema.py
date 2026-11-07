import glob, os
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"

def pages(channel):
    d = os.path.join(BASE, f"basic_{channel}.COMBINED")
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

for ch in ["player_hurt", "player_death", "round_end", "round_state", "weapon_fire"]:
    ps = pages(ch)
    print("==", ch, "pages:", len(ps))
    lf = pl.scan_parquet(ps)
    schema = lf.collect_schema()
    print("columns:", schema.names())
    print("dtypes:", {k: str(v) for k, v in schema.items()})
    df = lf.head(3).collect()
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(df)
