"""x4 without its armor columns (armor from hit events is only known for players who were hit, an era-specific selection)."""
from feat_x4_hp_detail import compute as _x4


def compute(G, store):
    df = _x4(G, store)
    return df.select([c for c in df.columns if "_arm_" not in c])
