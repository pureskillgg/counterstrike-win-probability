import glob, os, collections
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
p = glob.glob("G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds/basic_rank_update.2026-09-18,2026-10-08/dataframe_*")[0]
df = pq.read_table(p).to_pandas()
print("rows", len(df), "matches", df.match_key.nunique())
for c in ["player_phi_ang", "player_theta_ang", "player_z_vel", "player_x_vel", "player_y_vel"]:
    s = df[c].dropna()
    print(c, "n", len(s), "min", s.min(), "p01", s.quantile(.01), "p50", s.median(), "p99", s.quantile(.99), "max", s.max(), "pct_eq0", round((s == 0).mean()*100, 2))
print("z_vel top:", df.player_z_vel.value_counts(dropna=False).head(6).to_dict())
# null pattern per match
g = df.groupby("match_key").agg(n=("tick", "size"), nullpos=("player_x_pos", lambda s: s.isna().sum()))
print("matches all-null pos:", int((g.nullpos == g.n).sum()), "partial:", int(((g.nullpos > 0) & (g.nullpos < g.n)).sum()), "no-null:", int((g.nullpos == 0).sum()), "of", len(g))
print("null pos by player_id:", df.groupby("player_id").player_x_pos.apply(lambda s: round(s.isna().mean(), 3)).to_dict())
print("null pos by player_id_fixed null:", df.assign(fx=df.player_id_fixed.isna()).groupby("fx").player_x_pos.apply(lambda s: round(s.isna().mean(), 3)).to_dict())
print("null by round:", df.groupby("round").player_x_pos.apply(lambda s: round(s.isna().mean(), 2)).to_dict())
# player_id -> player_id_fixed mapping consistency
m = df.dropna(subset=["player_id_fixed"]).groupby(["match_key", "player_id"]).player_id_fixed.nunique()
print("match,player_id with >1 player_id_fixed value:", int((m > 1).sum()), "of", len(m))
pairs = df.dropna(subset=["player_id_fixed"])[["player_id", "player_id_fixed"]].value_counts().head(12)
print("top (player_id, player_id_fixed) pairs:\n", pairs.to_string())
# rank_change vs rank_new-rank_old
d = df.rank_new - df.rank_old
mis = df[d != df.rank_change]
print("mismatch rows", len(mis), "with rank_old==0:", int((mis.rank_old == 0).sum()), "rank_change==0 among mismatches:", int((mis.rank_change == 0).sum()))
print("mismatch rank_change value counts:", mis.rank_change.value_counts().head(8).to_dict())
print(mis[mis.rank_old != 0][["match_key", "round", "tick", "player_id", "rank_old", "rank_new", "rank_change", "win_count"]].head(8).to_string())
print("rows with rank_change -1000:")
print(df[df.rank_change == -1000][["match_key", "round", "tick", "player_id", "rank_old", "rank_new", "rank_change"]].head(4).to_string())
print("rank_change == -1 examples:")
print(df[df.rank_change == -1][["rank_old", "rank_new", "rank_change"]].value_counts().head(5).to_string())
print("rank_old==9999 or rank_new==9999:", int((df.rank_old == 9999).sum()), int((df.rank_new == 9999).sum()))
print("rank_old zero share:", round((df.rank_old == 0).mean(), 4), "rank_new zero share:", round((df.rank_new == 0).mean(), 4))
print("win_count vs rank: corr", round(df[["win_count", "rank_new"]].corr().iloc[0, 1], 3))
print("rank_new distribution by quantile:", df.rank_new.quantile([.1, .5, .9, .99]).to_dict())
print("rank_old>0 and rank_new==0 count:", int(((df.rank_old > 0) & (df.rank_new == 0)).sum()))
print("rank_old==rank_new count:", int((df.rank_old == df.rank_new).sum()))
