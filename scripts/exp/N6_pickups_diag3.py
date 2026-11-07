import glob, os
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
dr = in_clean(load("item_dropped", ["round","tick","player_id","def_index","reason","player_team_code"]))
dr = dr.filter(pl.col("tick") <= pl.col("end_tick"))
pk = in_clean(load("item_pickup", ["round","tick","player_id","def_index","player_team_code"]))
pk = pk.filter(pl.col("tick") <= pl.col("end_tick"))
rs = in_clean(load("round_start", ["round","tick"])).group_by(["midx","round"]).agg(pl.col("tick").min().alias("start_tick"))
ev = pl.concat([dr.select(["midx","round","tick","player_id","player_team_code"]), pk.select(["midx","round","tick","player_id","player_team_code"])])
ev = ev.filter(pl.col("player_team_code").is_in([2,3]))
per = ev.group_by(["midx","round","player_id"]).agg(pl.col("player_team_code").n_unique().alias("nt"), pl.len().alias("n"))
print("raw player-round groups with team codes:", per.height, " with >1 distinct team code:", per.filter(pl.col("nt")>1).height)
print("events in those multi-team groups:", int(per.filter(pl.col("nt")>1)["n"].sum()), "of", int(per["n"].sum()))
# per player-round: team code by ts? check whether team code differs between early and late events for same player
ev2 = ev.join(rs, on=["midx","round"], how="left").with_columns(((pl.col("tick")-pl.col("start_tick")).cast(pl.Float64)/TICK).alias("ts"))
first = ev2.group_by(["midx","round","player_id"]).agg(pl.col("player_team_code").first().alias("tc_mode"))
x = ev2.join(first, on=["midx","round","player_id"]).filter(pl.col("player_team_code")!=pl.col("tc_mode"))
print("events whose team differs from the player's first-seen team in that round:", x.height)
print("their ts pct 10/50/90:", [round(float(np.percentile(x["ts"].drop_nulls().to_numpy(),q)),1) for q in [10,50,90]] if x.height else None)
print("their round distribution:", x.group_by("round").agg(pl.len().alias("n")).sort("round").to_dicts())
# early enemy pairs by round
pk2 = pk.filter(pl.col("def_index").is_in(list(PRIM))).join(rs, on=["midx","round"], how="left").with_columns(((pl.col("tick")-pl.col("start_tick")).cast(pl.Float64)/TICK).alias("ts"))
dp = dr.filter(pl.col("def_index").is_in(list(PRIM))).select(["midx","round","def_index","tick","player_id","player_team_code","reason"])
fd = dp.group_by(["midx","round","def_index"]).agg(pl.col("tick").min().alias("ft"))
g = pk2.join(fd, on=["midx","round","def_index"], how="inner").filter(pl.col("ft")<pl.col("tick")).filter((pl.col("ts")<20)&(pl.col("round")>=2))
gj = g.join(dp, on=["midx","round","def_index"], how="inner", suffix="_d").filter(pl.col("tick_d")<pl.col("tick"))
gj = gj.with_columns(pl.when(pl.col("player_team_code")==pl.col("player_team_code_d")).then(pl.lit("same")).when(pl.col("player_team_code_d").is_in([2,3]) & pl.col("player_team_code").is_in([2,3])).then(pl.lit("enemy")).otherwise(pl.lit("unk")).alias("rel"))
print("early pick x earlier drop: relation by round band:")
print(gj.with_columns(pl.when(pl.col("round")<=12).then(pl.lit("r2-12")).otherwise(pl.lit("r13+")).alias("band")).group_by(["band","rel"]).agg(pl.len().alias("n")).sort(["band","rel"]))
print("early enemy pairs by round (top):", gj.filter(pl.col("rel")=="enemy").group_by("round").agg(pl.len().alias("n")).sort("n", descending=True).head(8).to_dicts())
