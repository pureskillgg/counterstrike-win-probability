"""Walking distance from the nearest living CT to the EXACT plant position over the navigation mesh (de_mirage, 2026, post-plant seconds).

This is the navigation version of the exact-plant margin in defuse_race_feature.py (the straight-line version). Per post-plant second
(stride 10, bomb planted, from data/combined/cache_snapshot_2026.parquet):
  ct_path_plant  = min over living CTs of the shortest mesh path from the CT's nav area to the plant's nav area

The plant position is the first recorded plant of the round (positions_1hz/_plants). The plant file has no z, so its nav area is the
nearest area by 2D centroid distance. A CT's area is find_area in 3D with nearest-centroid fallback, as in build_nav_path_features.py.
Walking distance is a Dijkstra over the area-centroid graph (edge weight = centroid-to-centroid distance); one run per plant area, cached.
Unreachable areas keep an infinite distance. Seconds with no living CT get no row (the eval fills them).

Writes data/diagnostics/nav_plant_de_mirage_stride10.parquet (match_key, round, second, ct_path_plant).
Run in the navigation environment (awpy 3.x):
  C:/Users/<user>/pureskill-nav312/Scripts/python.exe scripts/model/defuse_plant_nav.py
"""
import glob
import heapq
import os
import sys
import time

import numpy as np
import polars as pl
import awpy._awpy as core

ROOT = "G:/pureskill/csds-cs2"
DIAG = os.path.join(ROOT, "data", "diagnostics")
KEY = ["match_key", "round", "second"]
# Optional argument: the second stride (default 10). Stride 1 writes the every-second table for the stride check.
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = os.path.join(DIAG, f"nav_plant_de_mirage_stride{STRIDE}.parquet" if STRIDE != 10 else "nav_plant_de_mirage_stride10.parquet")
CT = 3


def keyed(df):
    return df.with_columns([pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)])


def main():
    t0 = time.time()
    nav = core.NavMesh(os.path.join(ROOT, "data", "geometry", "navs", "de_mirage.nav"))
    A = nav.areas.select(["area_id", "centroid_x", "centroid_y", "centroid_z"]).to_numpy()
    ids = A[:, 0].astype(int)
    C = A[:, 1:4].astype(float)
    index = {int(a): i for i, a in enumerate(ids)}
    nbr = [[index[int(n)] for n in nav.neighbors(int(a)) if int(n) in index] for a in ids]
    print(f"areas {len(ids)}, edges {sum(len(n) for n in nbr) // 2}", flush=True)

    def dijkstra(src):
        dist = np.full(len(ids), np.inf)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v in nbr[u]:
                nd = d + float(np.linalg.norm(C[u] - C[v]))
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    fields = {}

    def field(src):
        if src not in fields:
            fields[src] = dijkstra(src)
        return fields[src]

    cache = keyed(pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"),
                                  columns=["match_key", "round", "second", "map_name", "bomb_planted"]))
    post = (cache.filter((pl.col("map_name") == "de_mirage") & (pl.col("bomb_planted") == 1) & (pl.col("second") % STRIDE == 0))
                 .select(KEY).unique())
    print(f"post-plant stride-{STRIDE} keys: {post.height:,} ({time.time() - t0:.0f} s)", flush=True)

    plants = pl.concat([
        pl.read_parquet(f, columns=["match_key", "round", "x", "y"]).filter(pl.col("x").is_not_null() & pl.col("y").is_not_null())
          .group_by(["match_key", "round"]).agg(pl.col("x").first(), pl.col("y").first())
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "_plants", "2026-*.parquet")))])
    plants = plants.group_by(["match_key", "round"]).agg(pl.col("x").first(), pl.col("y").first())
    plants = plants.with_columns(pl.col("round").cast(pl.Int64))
    px = plants["x"].to_numpy()
    py = plants["y"].to_numpy()
    plant_idx = np.hypot(C[None, :, 0] - px[:, None], C[None, :, 1] - py[:, None]).argmin(axis=1)
    plants = plants.with_columns(pl.Series("plant_area", plant_idx.astype(np.int64))).select(["match_key", "round", "plant_area"])
    print(f"planted rounds with a plant position: {plants.height:,}", flush=True)

    rows = []
    fallback = 0
    total = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "2026-*.parquet"))):
        p = keyed(pl.read_parquet(f, columns=["match_key", "round", "second", "side", "x", "y", "z"]))
        p = p.filter(pl.col("side") == CT).join(post, on=KEY, how="inner").join(plants, on=["match_key", "round"], how="inner")
        if p.height == 0:
            continue
        xyz = p.select(["x", "y", "z"]).to_numpy()
        ct_area = np.empty(len(xyz), dtype=np.int64)
        for i, (x, y, z) in enumerate(xyz):
            a = nav.find_area((float(x), float(y), float(z)))
            total += 1
            if a is None:
                fallback += 1
                ct_area[i] = int(np.hypot(C[:, 0] - x, C[:, 1] - y).argmin())
            else:
                ct_area[i] = index[int(a)]
        pa = p["plant_area"].to_numpy()
        dist = np.empty(len(xyz), dtype=np.float64)
        for src in np.unique(pa):
            sel = pa == src
            dist[sel] = field(int(src))[ct_area[sel]]
        rows.append(p.select(KEY).with_columns(pl.Series("d", dist)))
        print(f"  {os.path.basename(f)}: {p.height:,} CT rows, {time.time() - t0:.0f} s", flush=True)

    R = pl.concat(rows)
    out = R.group_by(KEY).agg(pl.col("d").min().alias("ct_path_plant")).sort(KEY)
    n_inf = out.filter(pl.col("ct_path_plant").is_infinite()).height
    out.write_parquet(OUT)
    print(f"wrote {OUT}: {out.height:,} seconds, {n_inf:,} with an unreachable CT, "
          f"find_area fallback {fallback:,} of {total:,} CT rows, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
