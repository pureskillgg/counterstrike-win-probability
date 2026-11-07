"""The DM margin, as pinned in Meridian pin 56ae8e7d-dfe2-478d-87df-cda4ba027690 (A77+DM arm).

    margin = (40 - seconds_since_plant) - (c_min_bomb_dist / 250 + defuse_seconds)
    defuse_seconds = 5 if st_kit_n_ct >= 1 else 10
    NaN before the plant (seconds_since_plant is NaN there)

The 250 units per second is the same crude run speed as defuse_race_feature.py. Used by the margin check, the augmented
matrix build and the wrapper fit script, so all three compute the column the same way.
"""
import numpy as np

FUSE_S = 40.0
SPEED = 250.0


def margin_from_cols(seconds_since_plant, c_min_bomb_dist, st_kit_n_ct):
    sp = np.asarray(seconds_since_plant, dtype=np.float64)
    dist = np.asarray(c_min_bomb_dist, dtype=np.float64)
    kit = np.asarray(st_kit_n_ct, dtype=np.float64)
    defuse = np.where(kit >= 1, 5.0, 10.0)  # a NaN kit count takes the else branch (10 s); counted in the check
    m = (FUSE_S - sp) - (dist / SPEED + defuse)
    # seconds_since_plant is the sentinel -1 before the plant in this matrix, not NaN: mask it explicitly so the pre-plant NaN
    # rule does not depend on c_min_bomb_dist being NaN there. Planted rows have seconds_since_plant >= 0, so this changes no
    # finite value.
    m = np.where(sp < 0, np.nan, m)
    return m.astype(np.float32)
