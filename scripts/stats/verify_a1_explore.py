"""Independent verifier for A1 (armor analysis). Structure exploration only.

Checks how the first freeze-end snapshot per (match_key, round) is defined,
player counts per side, duplicates, and round numbering. Writes nothing
except stdout.
"""
import polars as pl

P = "G:/pureskill/csds-cs2/data/state_sample/players_1hz.parquet"
R = "G:/pureskill/csds-cs2/data/state_sample/rounds.parquet"

cols = ["match_key", "round", "tick", "seconds_since_live", "split", "player_id",
        "team_code", "armor", "has_helmet", "has_defuser", "equipment_value_calc"]
players = pl.scan_parquet(P).select(cols)
rounds = pl.scan_parquet(R).select(["match_key", "round", "round_number", "ct_win", "split"])

pm = players.select(pl.col("match_key").n_unique()).collect().item()
rm = rounds.select(pl.col("match_key").n_unique()).collect().item()
print("players distinct matches:", pm, " rounds distinct matches:", rm)

rr = rounds.collect()
print("rounds rows:", rr.height, " ct_win nulls:", rr["ct_win"].null_count())
print("ct_win value counts:", rr["ct_win"].value_counts().sort("ct_win").rows())
print("round == round_number for all rows:", bool((rr["round"] == rr["round_number"]).all()))
print("rounds per match: min/max", rr.group_by("match_key").len()["len"].min(),
      rr.group_by("match_key").len()["len"].max())
print("split counts (rounds):", rr["split"].value_counts().sort("split").rows())

pl_df = players.collect()
print("player rows:", pl_df.height)
print("team_code values:", pl_df["team_code"].value_counts().sort("team_code").rows())
print("split counts (players):", pl_df.select(pl.col("split").unique()).to_series().to_list())

# first snapshot = minimum tick within (match_key, round)
g = pl_df.group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("tick_min"),
    pl.col("tick").n_unique().alias("n_ticks"),
    pl.col("seconds_since_live").min().alias("ssl_min"),
    pl.col("seconds_since_live").max().alias("ssl_max"),
)
print("(match,round) groups in players:", g.height)
print("distinct ticks per group: value counts", g["n_ticks"].value_counts().sort("n_ticks").rows()[:10])
print("min seconds_since_live of groups: value counts (first 10)",
      g["ssl_min"].value_counts().sort("ssl_min").head(10).rows())
print("max seconds_since_live of groups: value counts (first 10)",
      g["ssl_max"].value_counts().sort("ssl_max").head(10).rows())

first = pl_df.join(g.select(["match_key", "round", "tick_min"]), on=["match_key", "round"]) \
    .filter(pl.col("tick") == pl.col("tick_min"))
dup = first.group_by(["match_key", "round", "player_id"]).len().filter(pl.col("len") > 1)
print("first-snapshot rows:", first.height, " duplicate (match,round,player):", dup.height)

cnt = first.group_by(["match_key", "round", "team_code"]).len() \
    .pivot(on="team_code", index=["match_key", "round"], values="len").fill_null(0)
print("pivot columns:", cnt.columns)
print("(CT,T) count pattern across all (match,round):")
ct_col = "3" if "3" in cnt.columns else 3
t_col = "2" if "2" in cnt.columns else 2
pat = cnt.group_by([ct_col, t_col]).len().sort("len", descending=True)
print(pat.head(20))

# round-number ranges of the rounds that appear in players
rnums = first.select("round").unique().sort("round")["round"].to_list()
print("rounds present (first snapshot):", len(rnums), "min", min(rnums), "max", max(rnums))
