"""Read-only check: does the cache field seconds_since_round_live (named in pin 64d46c76) agree with the round-state
second that gdt_weights uses for the pre-plant time third? Training-split rows only; no labels read."""
import numpy as np
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
CACHE = ROOT + "/data/combined/cache_snapshot_2026_ext.parquet"
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
TRAIN = 0

cs = pl.read_parquet_schema(CACHE)
print("cache has seconds_since_round_live:", "seconds_since_round_live" in cs,
      "dtype:", str(cs.get("seconds_since_round_live")))

frames = []
for m in MAPS:
    z = np.load(f"{ROOT}/data/round_state/{m}_ext.npz")
    sel = z["split"] == TRAIN
    mk = z["match_keys"].astype(str)[z["match_idx"][sel].astype(np.int64)]
    frames.append(pl.DataFrame({"match": mk, "round": z["round"][sel].astype(np.int64),
                                "second": z["second"][sel].astype(np.int64)}))
tr = pl.concat(frames)
keys = sorted(set(tr["match"].to_list()))
cache = (pl.scan_parquet(CACHE)
         .select(["match_key", "round", "second", "bomb_planted", "seconds_since_round_live"])
         .rename({"match_key": "match"})
         .filter(pl.col("match").is_in(keys))
         .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
         .collect())
j = tr.join(cache, on=["match", "round", "second"], how="inner")
print("train rows:", tr.height, " matched to cache:", j.height)
live = j["seconds_since_round_live"]
print("null seconds_since_round_live among matched:", int(live.null_count()))
d = j.filter(pl.col("seconds_since_round_live").is_not_null()).with_columns(
    (pl.col("seconds_since_round_live").cast(pl.Float64) - pl.col("second").cast(pl.Float64)).alias("diff"))
print("diff (live minus second) quantiles p0,p1,p50,p99,p100:",
      np.percentile(d["diff"].to_numpy(), [0, 1, 50, 99, 100]).round(3).tolist())
print("rows where live == second exactly:", int((d["diff"] == 0).sum()), "of", d.height)


def third(expr, length):
    return (pl.when(expr < length / 3).then(pl.lit("lot"))
            .when(expr < 2 * length / 3).then(pl.lit("medium")).otherwise(pl.lit("little")))


pre = d.filter(pl.col("bomb_planted") == 0)
b_second = pre.select(third(pl.col("second").cast(pl.Float64), 115.0).alias("b")).to_series()
b_live = pre.select(third(pl.col("seconds_since_round_live").cast(pl.Float64), 115.0).alias("b")).to_series()
print("pre-plant rows:", pre.height, " time-third differs (second vs live):", int((b_second != b_live).sum()))
print("pre-plant third share, second-based:", b_second.value_counts().sort("b").rows())
