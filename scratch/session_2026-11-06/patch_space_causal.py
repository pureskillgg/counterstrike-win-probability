"""Run ON the space (python3 - < this file). Applies the per-tick bomb-position lookup to the space's two copies.

Exact-match replacements, each old string must occur exactly once; a backup of each file is kept as *.pre_causal (first run only).
"""
import os

HOME = os.path.expanduser("~/adam/builder")
MP = HOME + "/posport/scripts/data/make_pos_features.py"
PB = HOME + "/pos_bucket.py"

MP_OLD1 = ('    KG = K.join(G, on=["match_key", "round", "tick"], how="left")\n'
           '    KB = KG.join(planted_xy, on=["match_key", "round"], how="left")\n')
MP_NEW1 = ('    KG = K.join(G, on=["match_key", "round", "tick"], how="left")\n'
           '    # bomb position per tick: the latest bomb_begin_plant of the same round at or before that tick (null before the first one).\n'
           '    # Backward asof on the tick, so a later plant can never reach an earlier second.\n'
           '    KB = K.sort("tick").join_asof(planted_xy.sort("tick"), on="tick", by=["match_key", "round"], strategy="backward").sort("kid")\n')
MP_OLD2 = ('    planted_xy = PL.sort("tick").group_by("match_key", "round", maintain_order=True).agg(pl.col("x").last().alias("bx"), '
           'pl.col("y").last().alias("by")).with_columns(pl.col("round").cast(pl.Int64))\n')
MP_NEW2 = ('    # one row per bomb_begin_plant event (tick, position), not one per round: the per-tick lookup in feats_for_day needs the tick\n'
           '    planted_xy = (PL.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).sort("tick", maintain_order=True)\n'
           '                  .unique(["match_key", "round", "tick"], keep="last", maintain_order=True)\n'
           '                  .select("match_key", "round", "tick", pl.col("x").alias("bx"), pl.col("y").alias("by")))\n')

PB_OLD1 = ('pxy = (plants.sort("tick").group_by("match_key", "round", maintain_order=True)\n'
           '       .agg(pl.col("x").last().alias("bx"), pl.col("y").last().alias("by")).with_columns(pl.col("round").cast(pl.Int64)))\n')
PB_NEW1 = ('# one row per bomb_begin_plant event (tick, position): feats_for_day looks the bomb up per tick (latest plant at or before it)\n'
           'pxy = (plants.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).sort("tick", maintain_order=True)\n'
           '       .unique(["match_key", "round", "tick"], keep="last", maintain_order=True)\n'
           '       .select("match_key", "round", "tick", pl.col("x").alias("bx"), pl.col("y").alias("by")))\n')
PB_OLD2 = 'OUTD = FIN + "/pos_parts"\n'
PB_NEW2 = 'OUTD = os.environ.get("POSB_OUT", FIN + "/pos_parts")  # POSB_OUT: write the chunks elsewhere (identity checks)\n'

PLAN = {MP: [(MP_OLD1, MP_NEW1), (MP_OLD2, MP_NEW2)], PB: [(PB_OLD1, PB_NEW1), (PB_OLD2, PB_NEW2)]}

for path, pairs in PLAN.items():
    text = open(path, encoding="utf-8").read()
    for old, new in pairs:
        n = text.count(old)
        assert n == 1, f"{path}: expected the old text once, found {n}: {old[:70]!r}"
        text = text.replace(old, new)
    bak = path + ".pre_causal"
    if not os.path.exists(bak):
        open(bak, "w", encoding="utf-8").write(open(path, encoding="utf-8").read())
    open(path, "w", encoding="utf-8").write(text)
    print("patched", path, "| backup", bak, flush=True)
print("ok", flush=True)
