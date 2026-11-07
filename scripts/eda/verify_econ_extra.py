"""Two follow-up checks for the economy and weapon notes.

1. rank_update vs bullet_damage: match-set overlap, by match_type.
2. molotov_fire density: is it per-tick? rows per (match, tick) and rows per fire.

Read-only. Run:  cd G:\\pureskill\\csds-cs2 ; pixi run python -I scripts\\eda\\verify_econ_extra.py
"""
import glob
import os

import pandas as pd
import pyarrow.parquet as pq

ROOT = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"


def read_ch(ch, cols):
    d = glob.glob(os.path.join(ROOT, "basic_%s.*" % ch))[0]
    frames = []
    for p in sorted(glob.glob(os.path.join(d, "dataframe_*"))):
        have = pq.ParquetFile(p).schema_arrow.names
        frames.append(pq.read_table(p, columns=[c for c in cols if c in have]).to_pandas())
    return pd.concat(frames, ignore_index=True)


hdr = pq.read_table(os.path.join(ROOT, "header", "dataframe_00000"),
                    columns=["key", "match_type"]).to_pandas()
mt = dict(zip(hdr.key, hdr.match_type))

rk = set(read_ch("rank_update", ["match_key"]).match_key)
bd = set(read_ch("bullet_damage", ["match_key"]).match_key)
print("rank_update %d, bullet_damage %d, both %d, rank only %d, bullet only %d" % (
    len(rk), len(bd), len(rk & bd), len(rk - bd), len(bd - rk)), flush=True)
for name, s in [("rank only", rk - bd), ("bullet only", bd - rk)]:
    print("  %s by match_type: %s" % (name, pd.Series([mt[k] for k in s]).value_counts().to_dict()), flush=True)

mf = read_ch("molotov_fire", ["match_key", "tick", "entity_id", "fire_index", "is_burning"])
print("molotov_fire rows %d | distinct (match,tick) %d | rows per (match,tick) median %s, max %d" % (
    len(mf), mf.groupby(["match_key", "tick"]).ngroups,
    mf.groupby(["match_key", "tick"]).size().median(), mf.groupby(["match_key", "tick"]).size().max()), flush=True)
per_fire = mf.groupby(["match_key", "entity_id", "fire_index"]).size()
print("molotov_fire distinct (match,entity_id,fire_index) %d | rows per fire median %s, max %d" % (
    len(per_fire), per_fire.median(), per_fire.max()), flush=True)
ib = mf.groupby(["match_key", "entity_id", "fire_index"]).is_burning.agg(["sum", "count"])
print("fires where is_burning is all True: %d, all False: %d, mixed: %d" % (
    int((ib["sum"] == ib["count"]).sum()), int((ib["sum"] == 0).sum()),
    int(((ib["sum"] > 0) & (ib["sum"] < ib["count"])).sum())), flush=True)
