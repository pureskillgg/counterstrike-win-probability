"""Portal-midpoint walking distances over the navigation mesh (de_mirage, 2026, every 10th second): the check on the centroid approximation.

Same rows and definitions as build_nav_path_features.py (CT and T nearest living player to each site, and the nearest living CT to the
exact plant position in defuse_plant_nav.py), but the path crosses each shared edge of the mesh at its midpoint, instead of joining area
centroids. The edge weight between neighbours u and v is |c_u - m_uv| + |m_uv - c_v|, where m_uv is the midpoint of the two shared corners
(awpy area(): corners, connections). Connections that share fewer than two corners fall back to the midpoint of the two centroids, which
reproduces the centroid weight exactly.

Writes data/diagnostics/nav_portal_de_mirage_stride10.parquet with columns
  match_key, round, second, ct_path_siteA_p, ct_path_siteB_p, t_path_siteA_p, t_path_siteB_p, ct_path_plant_p
Run in the navigation environment (awpy 3.x):
  C:/Users/13144/pureskill-nav312/Scripts/python.exe scripts/model/nav_portal_features.py
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
OUT = os.path.join(DIAG, "nav_portal_de_mirage_stride10.parquet")
sys.path.insert(0, ROOT + "/scripts/model")
sys.path.insert(0, ROOT + "/scripts/data")
import make_pos_features as MPF  # noqa: E402

KEY = ["match_key", "round", "second"]
STRIDE = 10
CT, T = 3, 2


def keyed(df):
    return df.with_columns([pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)])


def main():
    t0 = time.time()
    nav = core.NavMesh(os.path.join(ROOT, "data", "geometry", "navs", "de_mirage.nav"))
    det = {int(a): nav.area(int(a)) for a in nav.areas["area_id"].to_list()}
    ids = np.array(sorted(det))
    index = {int(a): i for i, a in enumerate(ids)}
    C = np.array([det[int(a)]["centroid"] for a in ids], dtype=float)
    corners = {int(a): set(tuple(round(float(c), 3) for c in p) for p in det[int(a)]["corners"]) for a in ids}

    nbr = [[] for _ in ids]
    wgt = [[] for _ in ids]
    n_portal = n_fallback = 0
    for a in ids:
        i = index[int(a)]
        for b in det[int(a)]["connections"]:
            b = int(b)
            if b not in index:
                continue
            j = index[b]
            shared = corners[int(a)] & corners[b]
            if len(shared) == 2:
                m = np.mean(np.array(list(shared), dtype=float), axis=0)
                n_portal += 1
            else:
                m = (C[i] + C[j]) / 2.0
                n_fallback += 1
            nbr[i].append(j)
            wgt[i].append(float(np.linalg.norm(C[i] - m) + np.linalg.norm(m - C[j])))
    print(f"areas {len(ids)}, directed connections {sum(len(n) for n in nbr)}: portal midpoints {n_portal}, centroid fallback {n_fallback} "
          f"({time.time() - t0:.0f} s)", flush=True)

    def dijkstra(src):
        dist = np.full(len(ids), np.inf)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in zip(nbr[u], wgt[u]):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    cache = {}

    def field(src):
        if src not in cache:
            cache[src] = dijkstra(src)
        return cache[src]

    centres = MPF.site_centres("combined")["de_mirage"]
    site_area = [int(np.hypot(C[:, 0] - c[0], C[:, 1] - c[1]).argmin()) for c in centres]
    site_fields = [field(s) for s in site_area]

    mt = pl.read_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet"), columns=["match_key", "map_name"])
    keys = mt.filter(pl.col("map_name") == "de_mirage")["match_key"].to_list()

    planted = pl.concat([
        pl.read_parquet(f, columns=["match_key", "round", "x", "y"]).filter(pl.col("x").is_not_null() & pl.col("y").is_not_null())
          .group_by(["match_key", "round"]).agg(pl.col("x").first(), pl.col("y").first())
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "_plants", "2026-*.parquet")))]).with_columns(
        pl.col("round").cast(pl.Int64))
    planted = planted.group_by(["match_key", "round"]).agg(pl.col("x").first(), pl.col("y").first())
    px, py = planted["x"].to_numpy(), planted["y"].to_numpy()
    plant_area = np.hypot(C[None, :, 0] - px[:, None], C[None, :, 1] - py[:, None]).argmin(axis=1)
    plant_of = {(mk, int(rd)): int(a) for mk, rd, a in zip(planted["match_key"].to_list(), planted["round"].to_list(), plant_area)}

    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "2026-*.parquet"))):
        p = keyed(pl.read_parquet(f, columns=["match_key", "round", "second", "side", "x", "y", "z"]))
        p = p.filter(pl.col("match_key").is_in(keys) & (pl.col("second") % STRIDE == 0) & pl.col("side").is_in([CT, T]))
        if p.height == 0:
            continue
        xyz = p.select(["x", "y", "z"]).to_numpy()
        area = np.empty(len(xyz), dtype=np.int64)
        for i, (x, y, z) in enumerate(xyz):
            a = nav.find_area((float(x), float(y), float(z)))
            area[i] = index[int(a)] if a is not None else int(np.hypot(C[:, 0] - x, C[:, 1] - y).argmin())
        side = p["side"].to_numpy()
        pA = site_fields[0][area]
        pB = site_fields[1][area]
        pP = np.full(len(xyz), np.nan)
        mk = p["match_key"].to_list()
        rd = p["round"].to_list()
        for i in np.where(side == CT)[0]:
            src = plant_of.get((mk[i], rd[i]))
            if src is not None:
                pP[i] = field(src)[area[i]]
        rows.append(p.select(KEY + ["side"]).with_columns(pl.Series("pA", pA), pl.Series("pB", pB), pl.Series("pP", pP)))
        print(f"  {os.path.basename(f)}: {p.height:,} rows, {time.time() - t0:.0f} s", flush=True)

    R = pl.concat(rows)
    agg = R.group_by(KEY + ["side"]).agg(pl.col("pA").min(), pl.col("pB").min(), pl.col("pP").min())
    ct = agg.filter(pl.col("side") == CT).select(KEY + [
        pl.col("pA").alias("ct_path_siteA_p"), pl.col("pB").alias("ct_path_siteB_p"), pl.col("pP").alias("ct_path_plant_p")])
    t = agg.filter(pl.col("side") == T).select(KEY + [
        pl.col("pA").alias("t_path_siteA_p"), pl.col("pB").alias("t_path_siteB_p")])
    out = ct.join(t, on=KEY, how="full", coalesce=True).sort(KEY)
    out.write_parquet(OUT)
    print(f"wrote {OUT}: {out.height:,} seconds, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
