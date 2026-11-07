import polars as pl
import glob, os
base = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
for ch in ["basic_round_end.COMBINED"]:
    d = os.path.join(base, ch)
    files = sorted(glob.glob(os.path.join(d, "dataframe_*")))
    print(ch, "n_files", len(files), files[:3])
    lf = pl.scan_parquet(files)
    print(lf.collect_schema())
    df = lf.collect()
    print("rows", df.height)
    print(df.head(3))
    print(df.select("match_key").unique().height, "distinct match_key")
    print(df["match_key"].str.slice(0, 13).value_counts())
    if "win_reason_message" in df.columns:
        print(df.group_by(["win_reason_code","win_reason_message"]).len().sort("win_reason_code"))
    if "player_count" in df.columns:
        print(df.group_by("player_count").len())
