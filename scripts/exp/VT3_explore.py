import polars as pl
import glob, os
import numpy as np
base = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
d = os.path.join(base, "basic_round_end.COMBINED")
files = sorted(glob.glob(os.path.join(d, "dataframe_*")))
re_ = pl.concat([pl.read_parquet(f) for f in files]).filter(pl.col("match_key").str.starts_with("csds/2026/"))
re_ = re_.filter(pl.col("player_count") == 10).select(["match_key","round","tick","winner_team_code","win_reason_code"]).rename({"tick":"re_tick"})
print("re10 rows", re_.height)

g = pl.scan_parquet(r"G:\pureskill\csds-cs2\data\viz\per_tick_states.parquet").collect()
print("grid rows", g.height, "match", g["match_key"].n_unique())
print("grid distinct (match,round)", g.select(["match_key","round"]).unique().height)
print("grid match prefix", g.select(pl.col("match_key").str.slice(0,10)).unique().to_series().to_list()[:10])
print("grid dup (match,tick):", g.height - g.select(["match_key","tick"]).unique().height)
print("grid dup (match,round,tick):", g.height - g.select(["match_key","round","tick"]).unique().height)
print("round vs round_number equal:", g.filter(pl.col("round")==pl.col("round_number")).height, "of", g.height)
print("sec min/max", g["seconds_since_round_live"].min(), g["seconds_since_round_live"].max())
print("sec non-integer count", g.filter((pl.col("seconds_since_round_live") - pl.col("seconds_since_round_live").round(0)).abs() > 1e-6).height)
print("bomb_planted dist", g.group_by("bomb_planted").len().to_dicts())
print("label dist", g.group_by("label_ct_win").len().to_dicts())
# join check
gr = g.select(["match_key","round"]).unique()
j = gr.join(re_, on=["match_key","round"], how="left")
print("grid rounds not in re10", j.filter(pl.col("re_tick").is_null()).height, "of", j.height)
print("re10 rounds not in grid", re_.join(gr, on=["match_key","round"], how="anti").height)
# label consistency
lab = g.select(["match_key","round","label_ct_win"]).unique(["match_key","round"])
lab = lab.join(re_, on=["match_key","round"], how="inner")
print("label vs winner mismatch", lab.filter(pl.col("label_ct_win") != (pl.col("winner_team_code")==3).cast(pl.Int64)).height, "of", lab.height)
# per-round max sec, min sec
per = g.group_by(["match_key","round"]).agg([
    pl.col("seconds_since_round_live").max().alias("maxsec"),
    pl.col("seconds_since_round_live").min().alias("minsec"),
    pl.len().alias("n"),
    pl.col("tick").max().alias("maxtick"),
    pl.col("tick").min().alias("mintick"),
    pl.col("label_ct_win").max().alias("lab"),
])
per = per.join(re_, on=["match_key","round"], how="left")
print("per-round minsec dist:", per.group_by("minsec").len().sort("minsec").head(10).to_dicts())
print("per-round n rows vs maxsec+1 mismatch:", per.filter(pl.col("n") != (pl.col("maxsec")-pl.col("minsec")+1)).height)
print("post-round grid rows (tick > re_tick):", per.with_columns((pl.col("maxtick")-pl.col("re_tick")).alias("d")).filter(pl.col("d") > 0).height)
print("maxtick - re_tick dist head:", per.with_columns((pl.col("maxtick")-pl.col("re_tick")).alias("d"))["d"].describe())
print("per-round maxsec dist describe", per["maxsec"].describe())
print(per.group_by("win_reason_code").agg([pl.len(), pl.col("maxsec").min().alias("minmax"), pl.col("maxsec").max().alias("maxmax"), pl.col("maxsec").median().alias("med")]).sort("win_reason_code"))
