import polars as pl
P = 'G:/pureskill/csds-cs2/data/state_sample/players_1hz.parquet'
R = 'G:/pureskill/csds-cs2/data/state_sample/rounds.parquet'
lf = pl.scan_parquet(P)
s0 = lf.filter(pl.col("seconds_since_live") == 0).collect()
print("sec0 rows", s0.height)
print("distinct inv_primary (sec0):", sorted(s0["inv_primary"].unique().to_list()))
print("distinct inv_primary (all rows):", sorted(lf.select(pl.col("inv_primary").unique()).collect()["inv_primary"].to_list()))
print("distinct weapon_code (all rows) count:", lf.select(pl.col("weapon_code").n_unique()).collect().item())
per_round = s0.group_by(["match_key","round"]).agg([
    pl.len().alias("n"),
    (pl.col("team_code")==2).sum().alias("nT"),
    (pl.col("team_code")==3).sum().alias("nCT"),
])
print("players per round distribution:")
print(per_round.group_by(["n","nT","nCT"]).agg(pl.len().alias("rounds")).sort(["n","nT","nCT"]))
print("rounds total:", per_round.height)
print("round number range per match:")
print(s0.group_by("match_key").agg([pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"), pl.col("round").n_unique().alias("nr")]).select([pl.col("rmin").value_counts().alias("x")]) if False else s0.group_by("match_key").agg(pl.col("round").n_unique().alias("nr")).group_by("nr").agg(pl.len().alias("matches")).sort("nr"))
pr = s0.filter(pl.col("round").is_in([1,13])).select(["match_key","round"]).unique()
print("pistol rounds (1 or 13) at sec0 in players set:", pr.height)
pr_clean = per_round.filter(pl.col("n")==10).filter(pl.col("round").is_in([1,13]))
print("pistol rounds in exact-10 clean set:", pr_clean.height)
print("round value counts (sec0):", s0.group_by("round").agg(pl.len().alias("n")).sort("round").head(30).write_csv(None))
rd = pl.read_parquet(R)
print("rounds.parquet rows:", rd.height, "splits:", rd.group_by("split").agg(pl.len()).to_dicts())
print(rd.head(3))
print("rounds.parquet alive_gap_snapshots value counts:")
print(rd.group_by("alive_gap_snapshots").agg(pl.len()).sort("alive_gap_snapshots").head(20))
