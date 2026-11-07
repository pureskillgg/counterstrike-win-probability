import glob, os, math
from collections import defaultdict
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
dr = in_clean(load("item_dropped", ["round","tick","item","def_index","reason","player_team_code"]))
dr = dr.filter(pl.col("tick") <= pl.col("end_tick"))
pk = in_clean(load("item_pickup", ["round","tick","item","def_index","player_team_code"]))
pk = pk.filter(pl.col("tick") <= pl.col("end_tick"))
rs = in_clean(load("round_start", ["round","tick"])).group_by(["midx","round"]).agg(pl.col("tick").min().alias("start_tick"))
pk = pk.join(rs, on=["midx","round"], how="left").with_columns(((pl.col("tick")-pl.col("start_tick")).cast(pl.Float64)/TICK).alias("ts"))
dr = dr.join(rs, on=["midx","round"], how="left").with_columns(((pl.col("tick")-pl.col("start_tick")).cast(pl.Float64)/TICK).alias("ts"))
fd = dr.group_by(["midx","round","def_index"]).agg(pl.col("tick").min().alias("first_drop_tick"))
pk = pk.join(fd, on=["midx","round","def_index"], how="left").with_columns((pl.col("first_drop_tick")<pl.col("tick")).fill_null(False).alias("ground"))
pk2 = pk.filter(pl.col("round")>=2)
pp = pk2.filter(pl.col("def_index").is_in(list(PRIM)))
print("== non-ground primary pickups r2+ t_since (s) percentiles")
ng = pp.filter(~pl.col("ground"))["ts"].drop_nulls().to_numpy()
print({q: round(float(np.percentile(ng,q)),2) for q in [1,10,50,90,99,99.9]}, "max", round(float(ng.max()),2), "n", len(ng))
print("== non-ground primary r2+ histogram 2s bins 0..60 (counts)")
h,_ = np.histogram(ng, bins=np.arange(0,62,2)); print(list(zip(np.arange(0,60,2).tolist(), h.tolist())))
g = pp.filter(pl.col("ground"))["ts"].drop_nulls().to_numpy()
print("== ground primary r2+ histogram 2s bins 0..60 (counts)")
h2,_ = np.histogram(g, bins=np.arange(0,62,2)); print(list(zip(np.arange(0,60,2).tolist(), h2.tolist())))
print("== ground primary r2+ n", len(g), "under 20s", int((g<20).sum()), "under 30s", int((g<30).sum()))
# primary death drops timing
pdd = dr.filter(pl.col("def_index").is_in(list(PRIM)))
for rsn in ["death","manual","replaced"]:
    x = pdd.filter(pl.col("reason")==rsn)["ts"].drop_nulls().to_numpy()
    print(f"== primary drops reason={rsn} r-all ts pct 10/50/90:", [round(float(np.percentile(x,q)),1) for q in [10,50,90]], "frac<20s", round(float((x<20).mean()),3), "n", len(x))
# link early ground pickups to drops by reason
early = pp.filter(pl.col("ground") & (pl.col("ts")<20)).select(["midx","round","def_index","tick","ts"])
print("early ground primary (r2+, ts<20) n:", early.height)
# build drop lookup for primary drops
dps = pdd.select(["midx","round","def_index","tick","ts","reason","player_team_code"])
j = early.join(dps, on=["midx","round","def_index"], how="left", suffix="_d").filter(pl.col("tick_d") < pl.col("tick"))
print("candidate (pick,drop) pairs with drop earlier:", j.height)
jj = j.group_by(["midx","round","def_index","tick"]).agg(pl.col("reason").first().alias("reason1"), pl.col("ts_d").min().alias("drop_ts_min"), pl.col("ts_d").max().alias("drop_ts_max"), pl.len().alias("ncand"))
print("reason of earliest-candidate for early pickups (count by reason):", jj.group_by("reason1").agg(pl.len().alias("n")).to_dicts())
print("drop ts for early pickups (min) pct:", [round(float(np.percentile(jj["drop_ts_min"].to_numpy(),q)),1) for q in [10,50,90]])
print("share of early pickups where a candidate drop also occurred before 20s:", round(float((jj["drop_ts_min"]<20).mean()),3))
# same-def drop during freeze, pickup in freeze
print("drops of primary in freeze (ts<20) by reason:", pdd.filter(pl.col("ts")<20).group_by("reason").agg(pl.len().alias("n")).to_dicts())
print("primary drops r2+ with ts<20 n", pdd.filter((pl.col("ts")<20)&(pl.col("round")>=2)).height, "total r2+", pdd.filter(pl.col("round")>=2).height)
