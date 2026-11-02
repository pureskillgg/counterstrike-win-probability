"""SELF-TEST ONLY: a deliberately leaky feature (whole-round death totals broadcast to every second). Must be REJECTED."""
import numpy as np
import polars as pl

from features_events import scan_events


def compute(G, store):
    d = scan_events(store, "player_death", ["match_key", "round", "tick", "player_id"])
    tot = d.group_by(["match_key", "round"]).agg(pl.len().alias("round_total_deaths"))   # includes deaths after the row's tick
    return G.select("match_key", "round").join(tot, on=["match_key", "round"], how="left").select(pl.col("round_total_deaths").cast(pl.Float32))
