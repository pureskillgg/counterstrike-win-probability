"""R4 probe (read-only): does the state sample keep each match on one side of its split?

Reads data/state_sample/rounds.parquet and positions_snapshots.parquet only. Writes nothing.
Checks:
  - matches that appear under more than one split label (rounds table and snapshot table);
  - ct_win base rate by split, at round level and at snapshot level;
  - snapshot count per round vs round length in ticks (how many states a round contributes);
  - rounds with no snapshots;
  - whether the snapshot split and ct_win agree with the rounds table for every round.
"""
import polars as pl

BASE = r"G:\pureskill\csds-cs2\data\state_sample"
rounds = pl.read_parquet(f"{BASE}\\rounds.parquet")
snap = pl.read_parquet(f"{BASE}\\positions_snapshots.parquet")

print("rounds:", rounds.height, "matches:", rounds["match_key"].n_unique())
print("snapshots:", snap.height, "rounds with snapshots:",
      snap.select(["match_key", "round"]).unique().height)

multi_r = (rounds.group_by("match_key").agg(pl.col("split").n_unique().alias("k"))
           .filter(pl.col("k") > 1))
multi_s = (snap.group_by("match_key").agg(pl.col("split").n_unique().alias("k"))
           .filter(pl.col("k") > 1))
print("matches on >1 split (rounds table):", multi_r.height)
print("matches on >1 split (snapshot table):", multi_s.height)

print("\nsplit sizes (rounds):")
print(rounds.group_by("split").agg(pl.len().alias("rounds"),
                                   pl.col("match_key").n_unique().alias("matches"),
                                   pl.col("ct_win").mean().alias("ct_win_rate")).sort("split"))

print("\nsplit sizes (snapshots):")
print(snap.group_by("split").agg(pl.len().alias("states"),
                                 pl.col("ct_win").mean().alias("ct_win_rate_states")).sort("split"))

per_round = (snap.group_by(["match_key", "round"])
             .agg(pl.len().alias("n_states"),
                  pl.col("tick").min().alias("tick_min"),
                  pl.col("tick").max().alias("tick_max"),
                  pl.col("ct_win").n_unique().alias("n_labels"),
                  pl.col("split").n_unique().alias("n_splits")))
print("\nlabels per round (should be 1):", per_round["n_labels"].value_counts().to_dicts())
print("splits per round (should be 1):", per_round["n_splits"].value_counts().to_dicts())

joined = rounds.join(per_round, on=["match_key", "round"], how="left")
missing = joined.filter(pl.col("n_states").is_null())
print("\nrounds with no snapshots:", missing.height)
if missing.height:
    print(missing.select(["match_key", "round", "split", "ct_win"]).head(5))

mismatch = joined.filter(pl.col("n_states").is_not_null() & (pl.col("split") != pl.col("n_splits").cast(pl.Utf8)))
both = joined.filter(pl.col("n_states").is_not_null())
print("rounds with snapshots:", both.height)

dur = both.with_columns(
    ((pl.col("round_end_tick") - pl.col("live_start_tick"))).alias("live_ticks"),
    (pl.col("tick_max") - pl.col("tick_min")).alias("snap_span"))
print("\nlive ticks per round:", dur["live_ticks"].describe())
print("snapshots per round:", dur["n_states"].describe())
print("correlation(n_states, live_ticks):",
      dur.select(pl.corr("n_states", "live_ticks")).item())
