import polars as pl, glob, os
T = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
def load(ch):
    files = sorted(glob.glob(f"{T}/basic_{ch}.COMBINED/dataframe_*"))
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed"), len(files)
for ch in ["round_end", "player_death", "round_state", "player_info", "player_personal", "player_hurt"]:
    df, nf = load(ch)
    print("=====", ch, "pages", nf, "rows", df.height)
    print(df.schema)
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(df.head(3))
    print("match_key prefixes:", df.select(pl.col("match_key").str.slice(0, 14).alias("p")).group_by("p").len().to_dicts())
