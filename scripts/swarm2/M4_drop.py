"""M4 drop check: why are matches dropped by map? Header is_wingman and round player_count by map (read-only)."""
import glob, os
import polars as pl
SCR = (r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
       r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M4")
T2026 = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
H = pl.concat([pl.read_parquet(f, columns=["key", "map_name", "is_wingman"]) for f in
               sorted(glob.glob(os.path.join(T2026, "header", "dataframe_*")))], how="vertical_relaxed")
kept = set(pl.read_parquet(os.path.join(SCR, "rounds_2026.parquet"))["match_key"].unique().to_list())
H = H.with_columns(pl.col("key").is_in(list(kept)).alias("kept"))
print("header matches by map x is_wingman, with kept count")
print(H.group_by(["map_name", "is_wingman"]).agg(pl.len().alias("n"), pl.col("kept").sum().alias("kept"))
      .sort(["map_name", "is_wingman"]))
RE = pl.concat([pl.read_parquet(f, columns=["match_key", "player_count"]) for f in
                sorted(glob.glob(os.path.join(T2026, "basic_round_end.2026-09-18,2026-10-08", "dataframe_*")))])
RE = RE.join(H.select(["key", "map_name", "is_wingman"]).rename({"key": "match_key"}), on="match_key", how="left")
pc = RE.filter(~pl.col("is_wingman").fill_null(False)).group_by(["map_name", "player_count"]).agg(pl.len().alias("rounds"))
print("\n5v5-header round_end rows by map x player_count (non-wingman), top values")
print(pc.sort(["map_name", "rounds"], descending=[False, True]).group_by("map_name", maintain_order=True).head(3))
