import os, sys
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import polars as pl
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(250)
base = {
 "2024": r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds\header\dataframe_00000",
 "2026": r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds\header\dataframe_00000",
}
for k, p in base.items():
    df = pl.read_parquet(p)
    print("=====", k, df.shape)
    print(df.schema)
    cols = [c for c in df.columns if c in ("match_date","server_name","build_num","demo_version_name","game_directory","game","platform","match_type","max_rounds","freeze_time","round_time_defuse","bomb_time","tick_save_rate","t_starters_score_final","ct_starters_score_final","number_of_points","max_unique_players","unique_steamids","fullpackets_version")]
    print(df.select(cols).head(3))
    if "match_date" in df.columns:
        print("match_date min/max:", df["match_date"].min(), df["match_date"].max())
    if "build_num" in df.columns:
        print(df["build_num"].value_counts().sort("build_num").head(40))
    if "game" in df.columns:
        print(df["game"].value_counts())
    if "game_directory" in df.columns:
        print(df["game_directory"].value_counts().head(20))
    if "server_name" in df.columns:
        print(df["server_name"].str.slice(0,40).value_counts().head(10))
