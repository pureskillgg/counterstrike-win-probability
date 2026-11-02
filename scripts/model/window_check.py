"""Share of alive coordinates outside the fixed round-state scaling window (for the 2024 rows against the 2026 de_mirage window).

A coordinate outside the window is clipped by the min-max scaling in embed_round_state, so a large share means the window is wrong for
the new rows. The check is reported per split and must be small before the window is kept fixed.

  out_of_window_share(x, y, z, window) -> float
  window: the 'window' entry of data/round_state/<map>.json: xy_min, xy_max (shared by x and y) and z_min, z_max.
"""
import numpy as np


def out_of_window_share(x, y, z, window):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if x.size == 0:
        return 0.0
    xy_lo, xy_hi = window["xy_min"], window["xy_max"]
    out = ((x < xy_lo) | (x > xy_hi) | (y < xy_lo) | (y > xy_hi)
           | (z < window["z_min"]) | (z > window["z_max"]))
    return float(out.mean())
