import polars as pl, glob
T = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
B = "G:/pureskill/csds-cs2/data"
def scan(ch, cols):
    files = sorted(glob.glob(f"{T}/basic_{ch}.COMBINED/dataframe_*"))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
pt = pl.read_parquet(f"{B}/viz/per_tick_states.parquet", columns=["match_key","round","tick","seconds_since_round_live"])
re = scan("round_end", ["match_key","round","tick","winner_team_code","player_count"]).filter(pl.col("match_key").str.starts_with("csds/2026/"))
de = scan("player_death", ["match_key","round","tick","player_id","player_team_code","attacker_id","weapon_name"]).filter(pl.col("match_key").str.starts_with("csds/2026/"))
pi = scan("player_info", ["match_key","round","player_id","team_code"]).filter(pl.col("match_key").str.starts_with("csds/2026/"))
# duplicates
dups = de.group_by(["match_key","round","player_id"]).len().filter(pl.col("len")>1)
print("dup keys:", dups.height, "max multiplicity:", dups["len"].max())
dd = de.join(dups.select(["match_key","round","player_id"]), on=["match_key","round","player_id"])
print(dd.sort(["match_key","round","player_id","tick"]).head(12))
# are dups same tick?
same_tick = dd.group_by(["match_key","round","player_id","tick"]).len().filter(pl.col("len")>1)
print("dup groups with identical tick:", same_tick.height, "of", dups.height)
# dups in the grid rounds?
gk = pt.select(["match_key","round"]).unique()
print("dup keys inside grid rounds:", dups.join(gk, on=["match_key","round"]).height)
# None victim team codes
none_rows = de.filter(pl.col("player_team_code").is_null())
print("null victim team rows:", none_rows.height, "in grid rounds:", none_rows.join(gk, on=["match_key","round"]).height)
print(none_rows.head(5))
# compare per-row victim team with player_info team (round-level roster)
m = de.join(pi, on=["match_key","round","player_id"], how="left")
print("death rows matched to player_info:", m.filter(pl.col("team_code").is_not_null()).height, "of", m.height)
agree = m.filter(pl.col("team_code").is_not_null() & pl.col("player_team_code").is_not_null())
print("victim team: player_team_code == player_info team_code:", agree.filter(pl.col("player_team_code")==pl.col("team_code")).height, "of", agree.height)
print("disagreements sample:")
print(agree.filter(pl.col("player_team_code")!=pl.col("team_code")).head(5))
# death window: live start (min tick with seconds_since_round_live==0 per round) and round_end tick
live = pt.filter(pl.col("seconds_since_round_live")==0).group_by(["match_key","round"]).agg(pl.col("tick").min().alias("live_tick"))
live_any = pt.group_by(["match_key","round"]).agg(pl.col("tick").min().alias("first_grid_tick"), pl.col("tick").max().alias("last_grid_tick"))
print("grid rounds with seconds==0 row:", live.height, "of", live_any.height)
w = live_any.join(re.select(["match_key","round","tick","winner_team_code","player_count"]).rename({"tick":"end_tick"}), on=["match_key","round"], how="left")
print("grid round: first grid tick vs live tick diff dist:")
w2 = w.join(live, on=["match_key","round"], how="left")
print((w2.select((pl.col("first_grid_tick")-pl.col("live_tick")).alias("d")).group_by("d").len().sort("len", descending=True).head(5)).to_dicts())
print("last grid tick <= end_tick:", w.filter(pl.col("last_grid_tick")<=pl.col("end_tick")).height, "of", w.height)
print("grid rounds with player_count==10:", w.filter(pl.col("player_count")==10).height)
# deaths relative to grid window
dw = de.join(live_any, on=["match_key","round"], how="inner").join(re.select(["match_key","round","tick"]).rename({"tick":"end_tick"}), on=["match_key","round"], how="left")
print("deaths in grid rounds:", dw.height)
print("death tick < live_tick:", dw.join(live, on=["match_key","round"]).filter(pl.col("tick") < pl.col("tick_right") if False else pl.col("tick") < pl.col("live_tick")).height)
print("death tick > end_tick:", dw.filter(pl.col("tick") > pl.col("end_tick")).height)
print("death tick within [first_grid_tick, end_tick]:", dw.filter((pl.col("tick")>=pl.col("first_grid_tick")) & (pl.col("tick")<=pl.col("end_tick"))).height)
