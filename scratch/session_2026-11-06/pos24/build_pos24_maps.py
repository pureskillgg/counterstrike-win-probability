"""Build the 24 train-only position inputs per map for the sequence screen (pin 66bca815, amended 7fe1d12b).

Source: data/combined/cache_pos_2026_trainctr.parquet (train-only site centres; rows for train and validation matches only),
joined to data/combined/cache_snapshot_2026.parquet on (match_key, round, tick) to get the second and the map.
Output: data/diagnostics/allmaps_pos24_trainsites/<map>.parquet with match_key, round, second, p0..p23.
The sequence script matches each state row to its row here by (match_key, round, second); any unmatched row stops the run.
"""
import os
import polars as pl

ROOT = "G:/pureskill/csds-cs2/"
OUT = ROOT + "data/diagnostics/allmaps_pos24_trainsites/"
os.makedirs(OUT, exist_ok=True)
POS = ["c_site_hi", "c_site_lo", "c_nearest_site_mean", "c_spread", "c_blind_n", "c_blind_sum",
       "t_site_hi", "t_site_lo", "t_nearest_site_mean", "t_spread", "t_blind_n", "t_blind_sum",
       "ct_t_centroid_dist", "min_enemy_dist", "c_in_contact", "t_in_contact", "c_sees_enemy", "t_sees_enemy",
       "c_exposed", "t_exposed", "c_min_bomb_dist", "c_n_bomb_500", "defuse_slack", "t_min_bomb_dist"]
assert len(POS) == 24
snap = pl.read_parquet(ROOT + "data/combined/cache_snapshot_2026.parquet",
                       columns=["match_key", "round", "tick", "second", "map_name"]).with_columns(
    pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("second").cast(pl.Int64)).unique(["match_key", "round", "tick"])
pos = pl.read_parquet(ROOT + "data/combined/cache_pos_2026_trainctr.parquet").with_columns(
    pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
j = pos.join(snap, on=["match_key", "round", "tick"], how="inner")
print("pos rows", pos.height, "joined", j.height, flush=True)
print("snapshot rows without position", snap.height - j.height, flush=True)
for m in sorted(j["map_name"].unique().to_list()):
    d = j.filter(pl.col("map_name") == m).select(["match_key", "round", "second"] + POS)
    d = d.rename({POS[i]: f"p{i}" for i in range(24)})
    dup = d.height - d.unique(["match_key", "round", "second"]).height
    assert dup == 0, (m, dup)
    d.write_parquet(OUT + f"{m}.parquet")
    print(m, d.height, "rows", flush=True)
print("POS24_DONE", flush=True)
