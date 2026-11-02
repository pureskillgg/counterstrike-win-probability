"""Distance-to-bomb features for de_mirage: walking distance over the navigation mesh AND straight-line distance (open data, awpy-data release 2000935, patch 1.41.9.2).

Per second (every 10th second, all de_mirage matches of the 2026 positions), for each site (two plant clusters from make_pos_features.site_centres):
  ct_path_site{A,B}  walking distance from the nearest living CT to the site (shortest path over the mesh, game units)
  t_path_site{A,B}   walking distance from the nearest living T to the site
  st_ct_site{A,B}    straight-line distance from the nearest living CT to the site centre (2D, game units)
  st_t_site{A,B}     straight-line distance from the nearest living T to the site centre
A player's area is the containing area (find_area); the positions on an area edge fall back to the nearest centroid (all within
100 game units). Unreachable areas (disconnected mesh pieces) keep an infinite distance; the eval clips them to a far value.

Runs in the navigation environment (Python 3.12 with awpy 3.x).
Writes data/diagnostics/nav_path_de_mirage_stride10.parquet and data/geometry/de_mirage_site_areas.json.
"""
import glob
import heapq
import json
import os
import sys
import time

import numpy as np
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
sys.path.insert(0, ROOT + "/scripts/model")
sys.path.insert(0, ROOT + "/scripts/data")
import awpy._awpy as core  # noqa: E402

STRIDE = 10
OUT = os.path.join(ROOT, "data", "diagnostics", "nav_path_de_mirage_stride10.parquet")
SITES_OUT = os.path.join(ROOT, "data", "geometry", "de_mirage_site_areas.json")
KINDS = ["ct_path", "t_path", "st_ct", "st_t"]


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

    import make_pos_features as MPF  # noqa: E402
    centres = MPF.site_centres("combined")["de_mirage"]
    site_area = [int(np.hypot(C[:, 0] - c[0], C[:, 1] - c[1]).argmin()) for c in centres]
    print("site centres", np.round(centres).tolist(), "-> areas", [int(ids[s]) for s in site_area], flush=True)
    fields = [dijkstra(s) for s in site_area]  # fields[k][i]: walking distance from area i to site k
    os.makedirs(os.path.dirname(SITES_OUT), exist_ok=True)
    with open(SITES_OUT, "w", encoding="utf-8") as f:
        json.dump({"site_centres_xy": np.round(centres, 2).tolist(), "site_area_ids": [int(ids[s]) for s in site_area]}, f, indent=1)

    mt = pl.read_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet"), columns=["match_key", "map_name"])
    keys = set(mt.filter(pl.col("map_name") == "de_mirage")["match_key"].to_list())
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "2026-*.parquet"))):
        p = pl.read_parquet(f, columns=["match_key", "round", "second", "side", "x", "y", "z"])
        p = p.filter(pl.col("match_key").is_in(list(keys)) & (pl.col("second") % STRIDE == 0))
        if p.height == 0:
            continue
        xyz = p.select(["x", "y", "z"]).to_numpy()
        area = np.empty(len(xyz), dtype=np.int64)
        for i, (x, y, z) in enumerate(xyz):
            a = nav.find_area((float(x), float(y), float(z)))
            if a is None:
                area[i] = int(np.hypot(C[:, 0] - x, C[:, 1] - y).argmin())
            else:
                area[i] = index[int(a)]
        dA, dB = fields[0][area], fields[1][area]
        sA = np.hypot(xyz[:, 0] - centres[0][0], xyz[:, 1] - centres[0][1])
        sB = np.hypot(xyz[:, 0] - centres[1][0], xyz[:, 1] - centres[1][1])
        rows.append(p.select(["match_key", "round", "second", "side"]).with_columns(
            pl.Series("pA", dA), pl.Series("pB", dB), pl.Series("sA", sA), pl.Series("sB", sB)))
        print(f"  {os.path.basename(f)}: {p.height} alive positions, {time.time() - t0:.0f} s", flush=True)

    R = pl.concat(rows)
    agg = R.group_by(["match_key", "round", "second", "side"]).agg(
        pl.col("pA").min(), pl.col("pB").min(), pl.col("sA").min(), pl.col("sB").min())
    out = agg.pivot(on="side", index=["match_key", "round", "second"], values=["pA", "pB", "sA", "sB"], aggregate_function="first")
    names = {"pA_3": "ct_path_siteA", "pB_3": "ct_path_siteB", "pA_2": "t_path_siteA", "pB_2": "t_path_siteB",
             "sA_3": "st_ct_siteA", "sB_3": "st_ct_siteB", "sA_2": "st_t_siteA", "sB_2": "st_t_siteB"}
    out = out.rename({k: v for k, v in names.items() if k in out.columns})
    for c in names.values():
        if c not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    out = out.select(["match_key", "round", "second"] + list(names.values()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.write_parquet(OUT)
    print(f"wrote {OUT}: {out.height} seconds, {out['match_key'].n_unique()} matches, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
