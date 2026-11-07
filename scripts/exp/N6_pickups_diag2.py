import glob, os, math
import numpy as np
import polars as pl
BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
TICK = 64.0
PRIM = {7,8,10,13,16,60,39,9,11,38,40,17,19,23,24,26,33,34}
def load(ch, cols):
    files = sorted(glob.glob(os.path.join(BASE, f"basic_{ch}.COMBINED", "dataframe_*")))
    lf = pl.scan_parquet(files).filter(pl.col("match_key").str.starts_with("csds/2026/"))
    return lf.select(["match_key"] + cols).collect()
rend = load("round_end", ["round","tick","player_count"])
clean0 = rend.filter(pl.col("player_count")==10).select(["match_key","round",pl.col("tick").alias("end_tick")])
mk = clean0.select("match_key").unique().sort("match_key").with_row_index("midx")
clean = clean0.join(mk, on="match_key")
def in_clean(df):
    return df.join(clean.select(["match_key","round","end_tick","midx"]), on=["match_key","round"], how="inner").drop("match_key")
dr = in_clean(load("item_dropped", ["round","tick","player_id","item","def_index","x_pos","y_pos","reason","player_team_code"]))
dr = dr.filter(pl.col("tick") <= pl.col("end_tick")).filter(pl.col("def_index").is_in(list(PRIM)))
pk = in_clean(load("item_pickup", ["round","tick","player_id","item","def_index","player_x_pos","player_y_pos","player_team_code"]))
pk = pk.filter(pl.col("tick") <= pl.col("end_tick")).filter(pl.col("def_index").is_in(list(PRIM)))
rs = in_clean(load("round_start", ["round","tick"])).group_by(["midx","round"]).agg(pl.col("tick").min().alias("start_tick"))
pk = pk.join(rs, on=["midx","round"], how="left").with_columns(((pl.col("tick")-pl.col("start_tick")).cast(pl.Float64)/TICK).alias("ts"))
dr = dr.join(rs, on=["midx","round"], how="left").with_columns(((pl.col("tick")-pl.col("start_tick")).cast(pl.Float64)/TICK).alias("dts"))
fd = dr.group_by(["midx","round","def_index"]).agg(pl.col("tick").min().alias("ft"))
pk = pk.join(fd, on=["midx","round","def_index"], how="left").with_columns((pl.col("ft")<pl.col("tick")).fill_null(False).alias("ground"))
g = pk.filter(pl.col("ground") & (pl.col("round")>=2)).with_columns((pl.col("ts")<20).alias("early"))
# team relation of pickup vs ANY earlier same-def drop in round (ignoring one-to-one), by reason
j = g.join(dr.select(["midx","round","def_index","tick","player_id","player_team_code","reason","x_pos","y_pos","dts"]),
           on=["midx","round","def_index"], how="inner", suffix="_d").filter(pl.col("tick_d")<pl.col("tick"))
j = j.with_columns(
    pl.when(pl.col("player_team_code")==pl.col("player_team_code_d")).then(pl.lit("same")).when(pl.col("player_team_code_d").is_in([2,3]) & pl.col("player_team_code").is_in([2,3])).then(pl.lit("enemy")).otherwise(pl.lit("unk")).alias("rel"),
    ((pl.col("tick")-pl.col("tick_d")).cast(pl.Float64)/TICK).alias("gap_s"))
print("pairs (pick x earlier same-def drop in round):", j.height)
print(j.group_by(["early","reason","rel"]).agg(pl.len().alias("n")).sort(["early","reason","rel"]))
# unique pickups per early flag and relation of their nearest-earlier drop
dd = j.with_columns(pl.col("gap_s").alias("g"))
print("gap_s pct (early pairs, all reasons):", [round(float(np.percentile(dd.filter(pl.col("early"))["g"].to_numpy(),q)),2) for q in [10,50,90]])
print("gap_s pct (late pairs, all reasons):", [round(float(np.percentile(dd.filter(~pl.col("early"))["g"].to_numpy(),q)),2) for q in [10,50,90]])
# distance pick vs drop, early manual pairs
e = j.filter(pl.col("early") & (pl.col("reason")=="manual") & pl.col("player_x_pos").is_not_null() & pl.col("x_pos").is_not_null())
dist = np.hypot((e["player_x_pos"]-e["x_pos"]).to_numpy(), (e["player_y_pos"]-e["y_pos"]).to_numpy())
print("early manual pairs distance pct 10/50/90:", [round(float(np.percentile(dist,q)),1) for q in [10,50,90]], "n", len(dist))
e2 = j.filter((~pl.col("early")) & (pl.col("reason")=="manual") & pl.col("player_x_pos").is_not_null() & pl.col("x_pos").is_not_null())
dist2 = np.hypot((e2["player_x_pos"]-e2["x_pos"]).to_numpy(), (e2["player_y_pos"]-e2["y_pos"]).to_numpy())
print("late manual pairs distance pct 10/50/90:", [round(float(np.percentile(dist2,q)),1) for q in [10,50,90]], "n", len(dist2))
# replaced: early vs late distance
r = j.filter(pl.col("early") & (pl.col("reason")=="replaced") & pl.col("player_x_pos").is_not_null() & pl.col("x_pos").is_not_null())
distr = np.hypot((r["player_x_pos"]-r["x_pos"]).to_numpy(), (r["player_y_pos"]-r["y_pos"]).to_numpy())
print("early replaced pairs distance pct 10/50/90:", [round(float(np.percentile(distr,q)),1) for q in [10,50,90]], "n", len(distr))
# does the same player (pick==drop id) occur in early pairs
print("early pairs self (same raw id):", j.filter(pl.col("early") & (pl.col("player_id")==pl.col("player_id_d"))).height)
