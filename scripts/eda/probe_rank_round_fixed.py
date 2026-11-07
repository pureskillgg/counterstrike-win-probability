import glob
import pandas as pd
import pyarrow.parquet as pq
p = glob.glob("G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds/basic_rank_update.2026-09-18,2026-10-08/dataframe_*")[0]
df = pq.read_table(p, columns=["match_key", "round", "tick", "player_id", "player_id_fixed", "rank_old", "rank_new", "rank_change", "win_count"]).to_pandas()
print("round value counts (top 12):", df["round"].value_counts().head(12).to_dict())
print("rows at round 24:", int((df["round"] == 24).sum()), "of", len(df))
per_m = df.groupby("match_key").agg(n_rounds=("round", "nunique"), n_ticks=("tick", "nunique"), n=("tick", "size"), rounds=("round", lambda s: sorted(set(s))))
print("matches with >1 round in rank_update:", int((per_m.n_rounds > 1).sum()), "of", len(per_m))
print("matches with >1 tick:", int((per_m.n_ticks > 1).sum()))
print("example multi-round matches:"); print(per_m[per_m.n_rounds > 1].head(4).to_string())
print("round-24 only matches:", int(((per_m.n_rounds == 1) & per_m.rounds.apply(lambda r: r == [24])).sum()))
# player_id_fixed bijection per match
g = df.dropna(subset=["player_id_fixed"]).groupby("match_key")
nf = g.player_id_fixed.nunique()
npid = g.player_id.nunique()
print("matches where distinct player_id_fixed != distinct player_id:", int((nf != npid).sum()), "of", len(nf))
pf = df.dropna(subset=["player_id_fixed"]).drop_duplicates(["match_key", "player_id", "player_id_fixed"])
dup_fixed = pf.groupby(["match_key", "player_id_fixed"]).player_id.nunique()
print("(match, player_id_fixed) shared by >1 player_id:", int((dup_fixed > 1).sum()))
print("player_id_fixed values per match (sample):", g.player_id_fixed.apply(lambda s: sorted(set(s.astype(int)))).head(3).to_dict())
print("player_id_fixed null rows by round:", df[df.player_id_fixed.isna()]["round"].value_counts().head(5).to_dict())
print("rank rows with rank_old==9999 detail:")
print(df[(df.rank_old == 9999) | (df.rank_new == 9999)][["match_key", "round", "player_id", "rank_old", "rank_new", "rank_change"]].to_string())
print("rank_change -100 example:", df[df.rank_change == -100][["rank_old", "rank_new", "rank_change"]].head(3).values.tolist())
print("count of exact -1000 by match:", df[df.rank_change == -1000].match_key.value_counts().head(5).to_dict())
