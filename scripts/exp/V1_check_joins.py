import polars as pl, glob
T = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
B = "G:/pureskill/csds-cs2/data"
def scan(ch, cols):
    files = sorted(glob.glob(f"{T}/basic_{ch}.COMBINED/dataframe_*"))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
pt = pl.read_parquet(f"{B}/viz/per_tick_states.parquet", columns=["match_key","round","tick","alive_ct","alive_t","label_ct_win","seconds_since_round_live"])
pr = pl.read_parquet(f"{B}/viz/per_tick_test_preds.parquet")
print("grid key prefixes:", pt.select(pl.col("match_key").str.slice(0,14).alias("p")).group_by("p").len().sort("p").to_dicts())
print("grid matches by year:", pt.select(pl.col("match_key").str.slice(5,4).alias("y"), "match_key").unique().group_by("y").len().to_dicts())
re = scan("round_end", ["match_key","round","tick","winner_team_code","win_reason_code","player_count","player_count_norm"])
print("round_end rows", re.height, "unique (mk,round):", re.select(["match_key","round"]).unique().height)
re26 = re.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("round_end 2026 rows", re26.height, "unique:", re26.select(["match_key","round"]).unique().height,
      "matches:", re26["match_key"].n_unique())
print("2026 player_count dist:", re26.group_by("player_count").len().sort("player_count").to_dicts())
print("2026 winner_team_code dist:", re26.group_by("winner_team_code").len().to_dicts())
print("2026 win_reason_code dist:", re26.group_by("win_reason_code").len().sort("len", descending=True).to_dicts())
# grid key membership
gk = pt.select(["match_key","round"]).unique()
print("grid unique rounds:", gk.height, " grid rounds in 2026 round_end:", gk.join(re26.select(["match_key","round"]).unique(), on=["match_key","round"], how="inner").height)
print("grid rounds 2026 keys:", gk.filter(pl.col("match_key").str.starts_with("csds/2026/")).height)
# dup check on round_end within (mk, round)
d = re26.group_by(["match_key","round"]).len().filter(pl.col("len")>1)
print("dup round_end (2026) keys:", d.height)
# label vs winner
j = gk.join(re26.unique(["match_key","round"], keep="first"), on=["match_key","round"], how="left")
lab = pt.group_by(["match_key","round"]).agg(pl.col("label_ct_win").n_unique().alias("nlab"), pl.col("label_ct_win").first().alias("label"))
jj = lab.join(re26.unique(["match_key","round"], keep="first"), on=["match_key","round"], how="inner")
print("label multi-valued within round:", lab.filter(pl.col("nlab")>1).height)
print("label vs winner crosstab (label, winner_team_code):")
print(jj.group_by(["label","winner_team_code"]).len().sort(["label","winner_team_code"]))
# death data
de = scan("player_death", ["match_key","round","tick","player_id","player_team_code","attacker_id","weapon_name"])
de26 = de.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("player_death 2026 rows", de26.height, "dup (mk,round,player_id):", de26.height - de26.select(["match_key","round","player_id"]).unique().height)
print("victim team code dist:", de26.group_by("player_team_code").len().to_dicts())
pi = scan("player_info", ["match_key","round","player_id","team_code"])
pi26 = pi.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("player_info 2026 rows", pi26.height, "unique (mk,round,pid):", pi26.select(["match_key","round","player_id"]).unique().height)
print("player_info team_code dist:", pi26.group_by("team_code").len().to_dicts())
per_round = pi26.group_by(["match_key","round"]).agg((pl.col("team_code")==3).sum().alias("ct"), (pl.col("team_code")==2).sum().alias("t"), pl.len().alias("n"))
print("players per round (n) dist:", per_round.group_by("n").len().to_dicts())
print("CT/T counts per round dist:", per_round.group_by(["ct","t"]).len().sort(["ct","t"]).to_dicts())
