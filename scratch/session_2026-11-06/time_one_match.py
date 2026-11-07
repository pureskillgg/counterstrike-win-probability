"""Time one bucket match on the space, split into: S3 downloads, parse + join, feature computation.

Read-only on the data. Imports pos_bucket with no chunk arguments, so its chunk loop does nothing.
"""
import os
import shutil
import sys
import time

sys.argv = ["timing"]
sys.path.insert(0, os.path.expanduser("~/adam/builder"))
import pos_bucket as PB  # noqa: E402
import make_pos_features as MPF  # noqa: E402
import polars as pl  # noqa: E402

mk = sorted(PB.bkeys & set(PB.kid))[0]
folder = mk.rsplit("/", 1)[0]
names = ["player_vector", "player_status", "player_info", "player_blind"]
print("match", mk, flush=True)

shutil.rmtree(f"{PB.SCR}/{folder}", ignore_errors=True)
t = time.time()
for n in names:
    PB.fetch(folder, n)
t_dl = time.time() - t
sizes = {n: os.path.getsize(f"{PB.SCR}/{folder}/{n}") for n in names}
print(f"S3 download, 4 files: {t_dl:.2f} s, {sum(sizes.values()) / 1e6:.1f} MB  {sizes}", flush=True)

t = time.time()
res = PB.one_match(mk)  # files are cached now: this is parse + join + wanted-seconds table only
t_parse = time.time() - t
assert res[0] == "ok", res
_, pos, sta, G, ev = res
print(f"parse + join: {t_parse:.2f} s; rows: pos {pos.height}, status {sta.height}, grid {G.height}", flush=True)

pxy_one = PB.pxy.filter(pl.col("match_key") == mk)
t = time.time()
f = MPF.feats_for_day(pos, sta, G, pxy_one, PB.sites, 2, ev)
t_feat = time.time() - t
print(f"feature computation: {t_feat:.2f} s; output rows {f.height}", flush=True)
print(f"total for this match: {t_dl + t_parse + t_feat:.2f} s (download share {100 * t_dl / (t_dl + t_parse + t_feat):.0f}%)", flush=True)
