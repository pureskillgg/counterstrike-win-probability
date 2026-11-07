import glob
import polars as pl
BASE = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds/"
SUF = ".2026-09-18,2026-10-08"
def read(ch, cols):
    folder = BASE + ("header" if ch == "header" else f"basic_{ch}{SUF}")
    fr = [pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(folder + "/dataframe_*")) if len(pl.read_parquet_schema(f)) > 0]
    return pl.concat(fr, how="vertical_relaxed")
H = read("header", ["key", "map_name", "max_rounds"]).rename({"key": "match_key"})
RE = read("round_end", ["match_key", "player_count"])
j = RE.join(H, on="match_key", how="left")
print("2026: header max_rounds x round player_count==4 (rounds), by map (top):")
print(j.with_columns((pl.col("player_count") == 4).alias("four")).group_by(["map_name", "max_rounds", "four"]).agg(pl.len().alias("rounds")).sort(["map_name", "max_rounds", "four"]).filter(pl.col("rounds") > 0).head(60))
w = j.filter(pl.col("player_count") == 4).select("match_key").unique()
print("distinct matches with any 4-player round:", w.height)
print("their max_rounds:", H.join(w, on="match_key").group_by("max_rounds").agg(pl.len().alias("matches")).sort("max_rounds").rows())
