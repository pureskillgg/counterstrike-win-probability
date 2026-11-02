"""Event-derived dynamic features for the per-second grid. All causal: a value at second tick ts uses only events with
tick <= ts inside the same round (information the model could have had live).

A feature module exposes  compute(G, store) -> polars DataFrame  with the same number of rows, in the same order as the grid
frame G (columns: match_key, round, tick, second, alive_ct, alive_t, bomb_planted, seconds_since_round_live,
seconds_since_plant, era, map_name, ...), holding ONLY the new columns.

Rule that makes the causality test possible: read events ONLY through scan_events(). The harness replaces the cutoff table
CUTOFF so that every round sees nothing after the row's tick, recomputes the feature, and rejects it if the value changes.

  hp_features(G, store)  hp_ct, hp_t   team hit points of the five players (500 = everyone untouched; a dead player counts 0).
                                       Built from player_hurt: per-player cumulative effective damage capped at 100, then capped
                                       at 100 per living player.
"""
import glob
import os

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
BIG = 1 << 31
T, CT = 2, 3
CUTOFF = None  # set by the harness: polars frame (match_key, round, cutoff_tick); events after cutoff_tick are hidden


def scan_events(store, ch, cols):
    """The only sanctioned event reader. Returns the channel's rows (all pages) for the given columns.
    Needs match_key, round and tick among cols whenever a cutoff is active."""
    d = os.path.join(ROOT, "tomes", store, "tome", "csds", f"basic_{ch}.COMBINED")
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(os.path.join(d, "dataframe_*")))],
                   how="vertical_relaxed")
    if CUTOFF is not None:
        df = df.join(CUTOFF, on=["match_key", "round"], how="inner")
        if "tick" in df.columns:  # static per-round tables (player_info) have no tick
            df = df.filter(pl.col("tick") <= pl.col("cutoff_tick"))
        df = df.drop("cutoff_tick")
    return df


def round_table(store):
    """Per-round bookkeeping from rounds.parquet: the freeze-end tick (live start) and the missing_side roster hint. Nothing
    derived from the outcome (final label, effective end) is exposed to feature code."""
    return pl.read_parquet(os.path.join(ROOT, "data", store, "rounds.parquet")).select(
        "match_key", "round", "freeze", "missing_side")


def _round_ids(G):
    keys = G.select("match_key", "round").unique(maintain_order=True).with_row_index("rid")
    return keys, G.join(keys, on=["match_key", "round"], how="left")["rid"].to_numpy().astype(np.int64)


def cum_by_round(G, ev, value_col, rid_g):
    """For each grid row, the sum of ev[value_col] over events of the same round with tick <= row tick.
    ev must carry rid (from _round_ids keys) and tick."""
    ev = ev.sort(["rid", "tick"])
    k = ev["rid"].to_numpy().astype(np.int64) * BIG + ev["tick"].to_numpy().astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(ev[value_col].to_numpy().astype(np.float64))])
    q = rid_g * BIG + G["tick"].to_numpy().astype(np.int64)
    base = rid_g * BIG
    return cs[np.searchsorted(k, q, side="right")] - cs[np.searchsorted(k, base, side="left")]


def hp_features(G, store="combined"):
    keys, rid_g = _round_ids(G)
    R = round_table(store)
    PI = (scan_events(store, "player_info", ["match_key", "round", "player_id", "team_code"])
          .unique(["match_key", "round", "player_id"], keep="last", maintain_order=True).rename({"team_code": "roster_team"}))
    H = scan_events(store, "player_hurt", ["match_key", "round", "tick", "player_id", "effective_health_removed", "player_team_code"])
    H = (H.join(keys, on=["match_key", "round"], how="inner")
         .join(R, on=["match_key", "round"], how="left")
         .filter(pl.col("tick") >= pl.col("freeze"))
         .join(PI, on=["match_key", "round", "player_id"], how="left")
         .with_columns(pl.coalesce([pl.col("player_team_code"), pl.col("roster_team"),
                                    pl.when(pl.col("roster_team").is_null()).then(pl.col("missing_side")).otherwise(None)]).alias("side"))
         .filter(pl.col("side").is_in([T, CT]))
         .sort(["rid", "player_id", "tick"])
         .with_columns(pl.col("effective_health_removed").clip(lower_bound=0).cum_sum().over(["rid", "player_id"]).alias("cum"))
         .with_columns(pl.col("cum").clip(upper_bound=100).alias("cum_c"))
         .with_columns((pl.col("cum_c") - pl.col("cum_c").shift(1).over(["rid", "player_id"]).fill_null(0)).alias("inc")))
    out = {}
    for side, name in ((CT, "hp_ct"), (T, "hp_t")):
        deficit = cum_by_round(G, H.filter(pl.col("side") == side), "inc", rid_g)
        alive = G["alive_ct" if side == CT else "alive_t"].to_numpy().astype(np.int64)
        # a player killed with a few HP of recorded damage missing would leave phantom HP: cap at 100 per living player
        out[name] = np.minimum(500 - deficit, 100 * alive).astype(np.float32)
    return pl.DataFrame(out)


compute = hp_features  # the module doubles as a feature module (name used by the harness)
