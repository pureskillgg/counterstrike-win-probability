# A2: how many round_end (E, F-convention) labels have no round_start (S) row with the same label, on tomes/2024-02-g.
# Mirrors bakeoff.py: k = (match, round_end label); kept only if (match, k) in round_start keys.
import os, sys, json, collections
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
re_ = ch_frame("round_end", ["round", "match_key"]).unique(["match_key", "round"]).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
rs = ch_frame("round_start", ["round", "match_key"]).unique(["match_key", "round"])
j = re_.join(rs.with_columns(pl.lit(1).alias("has_s")), on=["match_key", "round"], how="left")
miss = j.filter(pl.col("has_s").is_null())
print("round_end (match,label) pairs:", re_.height, " without round_start row with same label:", miss.height, flush=True)
print("  by group:", miss.with_columns(pl.col("day").map_elements(lambda d: "01" if d == "01" else ("02_07" if d <= "07" else "12_25"), return_dtype=pl.Utf8).alias("g")).group_by("g").len().sort("g").rows(), flush=True)
pm = miss.group_by("match_key").agg(pl.col("round").alias("labels"))
print("  labels missing per match (sample):", pm.head(5).rows(), flush=True)
print("  rank of missing labels relative to match max:", collections.Counter(miss.join(re_.group_by("match_key").agg(pl.col("round").max().alias("mx")), on="match_key").with_columns((pl.col("round") == pl.col("mx")).alias("is_last"))["is_last"].to_list()), flush=True)
sm = rs.join(re_.with_columns(pl.lit(1).alias("has_e")), on=["match_key", "round"], how="left").filter(pl.col("has_e").is_null())
print("round_start (match,label) pairs with no round_end same label:", sm.height, flush=True)
print("DONE", flush=True)
