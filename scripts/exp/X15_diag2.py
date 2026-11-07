import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import polars as pl
SCR = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad/"
dd = pl.read_parquet(SCR + "X15_hero_dd.parquet")
dd = dd.with_columns((pl.col("side_ra_t1") == 0).alias("no_rifle_after"))
# victim-side win probability at t (signed so higher = better for victim side)
dd = dd.with_columns([
    pl.when(pl.col("team_code") == 3).then(pl.col("p_c_t")).otherwise(1 - pl.col("p_c_t")).alias("vside_wp_t"),
    pl.when(pl.col("team_code") == 3).then(pl.col("p_c_t1")).otherwise(1 - pl.col("p_c_t1")).alias("vside_wp_t1"),
])
tab = (dd.group_by(["victim_class", "no_rifle_after"])
       .agg(pl.len().alias("n"), pl.col("vside_wp_t").mean().alias("vside_wp_t"), pl.col("vside_wp_t1").mean().alias("vside_wp_t1"),
            pl.col("drop_a").mean().alias("drop_a"), pl.col("drop_c").mean().alias("drop_c"))
       .sort(["victim_class", "no_rifle_after"]))
print(tab)
