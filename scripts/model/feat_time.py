"""Time features aimed at the end of the round (N3: the model underpredicts CT in the last seconds before the 115 s clock runs out).

Uses only the grid row's own columns (seconds_since_round_live, bomb_planted), so it is causal by construction.
  t_unplanted      seconds since live while the bomb is NOT planted, NaN once planted. Carries a monotone (+1) constraint:
                   with the bomb down, more elapsed time never lowers CT's chance, and the clock expiry is a CT win.
  secs_left        115 - seconds_since_round_live while unplanted (NaN after the plant): time until a CT win by the clock.
"""
import numpy as np
import polars as pl

ROUND_SECONDS = 115.0


def compute(G, store=None):
    t = G["seconds_since_round_live"].to_numpy().astype(np.float64)
    planted = G["bomb_planted"].to_numpy().astype(np.int64) == 1
    tu = np.where(planted, np.nan, t)
    return pl.DataFrame({"t_unplanted": tu.astype(np.float32), "secs_left": (ROUND_SECONDS - tu).astype(np.float32)})
