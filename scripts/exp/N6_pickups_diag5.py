import glob, os
import numpy as np
import polars as pl
BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
PRIM = {7,8,10,13,16,60,39,9,11,38,40,17,19,23,24,26,33,34}
def load(ch, cols):
    files = sorted(glob.glob(os.path.join(BASE, f"basic_{ch}.COMBINED", "dataframe_*")))
    lf = pl.scan_parquet(files).filter(pl.col("match_key").str.starts_with("csds/2026/"))
    return lf.select(["match_key"] + cols).collect()
rend = load("round_end", ["round","tick","player_count"])
print("round_end player_count by round (share ==10):")
t = rend.with_columns((pl.col("player_count")==10).alias("c10")).group_by("round").agg(pl.col("c10").mean().alias("share10"), pl.len().alias("n")).sort("round")
print(t.head(4).to_dicts())
clean0 = rend.filter(pl.col("player_count")==10).select(["match_key","round",pl.col("tick").alias("end_tick")])
print("clean rounds per round number (1..4):", clean0.group_by("round").agg(pl.len().alias("n")).sort("round").head(4).to_dicts())
dr = load("item_dropped", ["round","tick","def_index","reason"])
pk = load("item_pickup", ["round","tick","def_index"])
for rn in [1,2]:
    d = dr.filter((pl.col("round")==rn) & pl.col("def_index").is_in(list(PRIM)))
    p = pk.filter((pl.col("round")==rn) & pl.col("def_index").is_in(list(PRIM)))
    print(f"round {rn}: primary drops(all matches)={d.height}, primary pickups={p.height}")
    if d.height: print("   drop tick pct 5/50/95:", [int(np.percentile(d['tick'].to_numpy(),q)) for q in [5,50,95]], "reasons", d.group_by('reason').agg(pl.len()).to_dicts())
    if p.height: print("   pickup tick pct 5/50/95:", [int(np.percentile(p['tick'].to_numpy(),q)) for q in [5,50,95]])
