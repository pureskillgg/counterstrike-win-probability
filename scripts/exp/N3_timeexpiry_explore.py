import os, glob
import numpy as np
import polars as pl
pl.Config.set_tbl_rows(40)
TOME = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds\basic_round_end.COMBINED"
pages = sorted(glob.glob(os.path.join(TOME, "dataframe_*")))
re_all = pl.concat([pl.read_parquet(p) for p in pages], how="diagonal_relaxed")
re26 = (re_all.filter(pl.col("match_key").str.starts_with("csds/2026/"))
        .select(["match_key","round","tick","second","winner_team_code","win_reason_code","win_reason_message","player_count"]))
re10 = re26.filter(pl.col("player_count") == 10)
print("re10 rows", re10.height, "unique (match,round)", re10.select(["match_key","round"]).unique().height)
dup = re10.group_by(["match_key","round"]).agg(pl.len().alias("n")).filter(pl.col("n")>1)
print("dup (match,round) in re10:", dup.height)
print("re10 reason 12 rows:", re10.filter(pl.col("win_reason_code")==12).height)
print("re10 reason x winner:", re10.group_by(["win_reason_code","winner_team_code"]).agg(pl.len().alias("n")).sort("win_reason_code").to_dicts())
print("re10 reason 12 round_end.second quantiles:")
print(re10.filter(pl.col("win_reason_code")==12).select(pl.col("second").quantile(0.05).alias("q05"), pl.col("second").median().alias("med"), pl.col("second").quantile(0.95).alias("q95"), pl.col("second").min().alias("min"), pl.col("second").max().alias("max")))
# match date range from keys
mk = re10.select(pl.col("match_key").str.slice(5, 10).alias("date")).unique().sort("date")
print("date prefixes in re10:", mk.to_series().to_list())

g = pl.scan_parquet(r"G:\pureskill\csds-cs2\data\viz\per_tick_states.parquet").collect()
print("grid rows", g.height, "matches", g["match_key"].n_unique(), "rounds", g.select(["match_key","round"]).unique().height)
print("grid date prefixes:", g.select(pl.col("match_key").str.slice(5,10).alias("d")).unique().sort("d")["d"].to_list())
print("grid sec dtype, min, max:", g["seconds_since_round_live"].min(), g["seconds_since_round_live"].max())
# non-integer seconds?
frac = g.filter((pl.col("seconds_since_round_live") - pl.col("seconds_since_round_live").round(0)).abs() > 1e-9).height
print("non-integer sec rows:", frac)
# per-round last second
lastsec = g.group_by(["match_key","round"]).agg([
    pl.col("seconds_since_round_live").max().alias("last_sec"),
    pl.col("seconds_since_round_live").min().alias("first_sec"),
    pl.len().alias("n_rows"),
    pl.col("tick").min().alias("tick_min"),
    pl.col("tick").max().alias("tick_max"),
    pl.col("label_ct_win").max().alias("lab_max"), pl.col("label_ct_win").min().alias("lab_min"),
    pl.col("bomb_planted").max().alias("planted_any"),
])
print("rounds in grid:", lastsec.height)
print("first_sec value counts:", lastsec.group_by("first_sec").agg(pl.len().alias("n")).sort("first_sec").head(5).to_dicts())
print("label consistent within round:", int((lastsec["lab_max"] != lastsec["lab_min"]).sum()), "rounds inconsistent")
print("last_sec quantiles:", lastsec.select([pl.col("last_sec").quantile(q).alias(f"q{int(q*100)}") for q in [0.0,0.05,0.25,0.5,0.75,0.95,1.0]]).to_dicts())
print("last_sec histogram (1s):")
print(lastsec.group_by(pl.col("last_sec").floor()).agg(pl.len().alias("n")).sort("last_sec").to_dicts())
j = lastsec.join(re10.select(["match_key","round","win_reason_code","winner_team_code","tick","second"]), on=["match_key","round"], how="left")
print("grid rounds with re10 match:", j.filter(pl.col("win_reason_code").is_not_null()).height, "of", j.height)
print("grid rounds w/o re10:", j.filter(pl.col("win_reason_code").is_null()).height)
print("reason 12 grid rounds:", j.filter(pl.col("win_reason_code")==12).height)
r12 = j.filter(pl.col("win_reason_code")==12)
print("reason12 last_sec quantiles:", r12.select([pl.col("last_sec").quantile(q).alias(f"q{int(q*100)}") for q in [0.0,0.05,0.25,0.5,0.75,0.95,1.0]]).to_dicts())
print("reason12 last_sec value counts (top):", r12.group_by("last_sec").agg(pl.len().alias("n")).sort("n", descending=True).head(8).to_dicts())
print("reason12 round_end tick - tick_max:", (r12["tick"] - r12["tick_max"]).describe())
print("label vs winner_team_code mismatch:", j.filter(pl.col("winner_team_code").is_not_null() & (((pl.col("winner_team_code")==3).cast(pl.Int64)) != pl.col("lab_max"))).height)
# planted monotone?
mono = (g.sort(["match_key","round","tick"])
        .with_columns(pl.col("bomb_planted").shift(1).over(["match_key","round"]).alias("prev"))
        .filter(pl.col("prev").is_not_null() & (pl.col("bomb_planted") < pl.col("prev"))))
print("planted decreasing transitions:", mono.height)
print("round ends with planted rows: counts of planted_any:", lastsec.group_by("planted_any").agg(pl.len().alias("n")).to_dicts())
# seconds at which planted first; ticks
print("unplanted rounds with last_sec >=110:", lastsec.filter((pl.col("last_sec")>=110)).height)
print("ct_alive_minus_t values:", g.group_by("ct_alive_minus_t").agg(pl.len().alias("n")).sort("ct_alive_minus_t").to_dicts())
print("sample tick diffs:", g.sort(["match_key","round","tick"]).select((pl.col("tick").diff().over(["match_key","round"])).value_counts().sort("count", descending=True).head(5) if False else pl.lit(None)).head(1).to_dicts())
