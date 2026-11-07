"""Does the verified parity data carry bomb-position values before the first plant of a round? Read-only."""
import polars as pl

FIN = "/home/sagemaker-user/adam/builder/final"
P = pl.read_parquet(FIN + "/plants_all.parquet")
print("plants_all schema:", {k: str(v) for k, v in P.schema.items()}, "rows", P.height, flush=True)
first = (P.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
         .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("plant_tick")))
print("rounds with a plant event in plants_all:", first.height, flush=True)
feat = ["c_min_bomb_dist", "t_min_bomb_dist", "c_n_bomb_500", "defuse_slack"]
for c in ["c050", "c060"]:
    d = pl.read_parquet(f"{FIN}/pos_parity/bucket_{c}.parquet", columns=["match_key", "round", "tick"] + feat)
    d = d.join(first, on=["match_key", "round"], how="left")
    withp = d.filter(pl.col("plant_tick").is_not_null())
    pre = withp.filter(pl.col("tick") < pl.col("plant_tick"))
    post = withp.filter(pl.col("tick") >= pl.col("plant_tick"))
    noplant = d.filter(pl.col("plant_tick").is_null())
    print(f"\n== bucket_{c}: rows {d.height} | rows in rounds with a plant {withp.height} | "
          f"pre-plant rows {pre.height} | post-plant rows {post.height} | rows in rounds without a plant {noplant.height}", flush=True)
    for f in feat:
        print(f"  {f:18s} non-null pre-plant {pre.filter(pl.col(f).is_not_null()).height:9d} | "
              f"non-null post-plant {post.filter(pl.col(f).is_not_null()).height:9d} | "
              f"non-null no-plant-round {noplant.filter(pl.col(f).is_not_null()).height:9d}", flush=True)
