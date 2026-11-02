"""Walking distances over the navigation mesh for every living player at every second, for one map's extended round-state table.

Per row (one second of one round) and per map, four features, the same definitions as the de_mirage straight-line features in
train_temporal_allmaps.py --dist, but along the mesh:
  ct_A, t_A, ct_B, t_B   walking distance from the nearest living CT (or T) to each of the map's two site areas, in game units / 1000,
                         capped at 10 (10 when that side has no living player or the area is unreachable)

Method, as in build_nav_path_features.py (de_mirage) so the maps are comparable:
  - a player's area is find_area on its 3D position (positions are recovered exactly from the scaled round-state coordinates with the
    map's window); a player on no area falls back to the nearest area centroid (2D)
  - walking distance is a Dijkstra over the area-centroid graph (edge weight = centroid-to-centroid distance); one run per site area
  - site areas are the areas nearest to the map's two site centres (make_pos_features.site_centres)

Writes data/round_state/<map>_ext_navwalk.npy, float32, shape (N, 4), aligned with the rows of <map>_ext.npz.
Run in the navigation environment (awpy 3.x, scipy):
  C:/Users/<user>/pureskill-nav312/Scripts/python.exe scripts/model/build_nav_walk_allmaps.py <map> [<map> ...]
Set NAVWALK_MAX_ROWS to test on a prefix of the rows.
"""
import heapq
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import awpy._awpy as core

ROOT = "G:/pureskill/csds-cs2"
DATA = os.path.join(ROOT, "data", "round_state")
GEOM = os.path.join(ROOT, "data", "geometry", "navs")
MISSING = 1.0e4
CHUNK = 250_000
sys.path.insert(0, os.path.join(ROOT, "scripts", "data"))


def build(m):
    t0 = time.time()
    nav = core.NavMesh(os.path.join(GEOM, f"{m}.nav"))
    A = nav.areas.select(["area_id", "centroid_x", "centroid_y", "centroid_z"]).to_numpy()
    ids = A[:, 0].astype(int)
    C = A[:, 1:4].astype(float)
    index = {int(a): i for i, a in enumerate(ids)}
    nbr = [[] for _ in ids]
    for a in ids:
        d = nav.area(int(a))
        i = index[int(a)]
        for b in d["connections"]:
            b = int(b)
            if b in index and b != int(a):
                nbr[i].append(index[b])
    w = [[float(np.linalg.norm(C[i] - C[j])) for j in nbr[i]] for i in range(len(ids))]

    def dijkstra(src):
        dist = np.full(len(ids), np.inf)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, wt in zip(nbr[u], w[u]):
                nd = d + wt
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    import make_pos_features as MPF  # noqa: E402  (prints its own site table on import)
    if os.environ.get("NAVWALK_TRAIN_CENTRES", "0") == "1":
        # site centres fitted on TRAIN matches only (the leak-free version; see the sitecheck notes)
        man = json.load(open(os.path.join(ROOT, "data", "combined", "split_manifest.json"), encoding="utf-8"))["assign"]
        train_keys = [k for k, v in man.items() if v == "train"]
        centres = MPF.site_centres("combined", match_keys=train_keys)[m]
    else:
        centres = MPF.site_centres("combined")[m]
    site_idx = [int(np.hypot(C[:, 0] - c[0], C[:, 1] - c[1]).argmin()) for c in centres]
    fields = [dijkstra(s) for s in site_idx]
    print(f"{m}: {len(ids)} areas, site areas {[int(ids[s]) for s in site_idx]} ({time.time() - t0:.0f} s)", flush=True)

    win = json.load(open(os.path.join(DATA, f"{m}_ext.json"), encoding="utf-8"))["window"]
    lo, span = win["xy_min"], win["xy_max"] - win["xy_min"]
    zlo, zspan = win["z_min"], win["z_max"] - win["z_min"]
    X = np.load(os.path.join(DATA, f"{m}_ext.npz"))["X"]
    cap = int(os.environ.get("NAVWALK_MAX_ROWS", "0") or 0)
    N = X.shape[0] if cap <= 0 else min(cap, X.shape[0])
    tree = cKDTree(C[:, :2])
    out = np.full((N, 4), 10.0, dtype=np.float32)
    n_fb = n_look = 0
    for c in range(0, N, CHUNK):
        e = min(N, c + CHUNK)
        blk = np.asarray(X[c:e])
        alive = blk[:, :, 0] > 0.5
        rr, ss = np.nonzero(alive)
        xs = blk[rr, ss, 1].astype(np.float64) * span + lo
        ys = blk[rr, ss, 2].astype(np.float64) * span + lo
        zs = blk[rr, ss, 3].astype(np.float64) * zspan + zlo
        area = np.full(rr.size, -1, dtype=np.int64)
        for j, (x, y, z) in enumerate(zip(xs.tolist(), ys.tolist(), zs.tolist())):
            a = nav.find_area((x, y, z))
            if a is not None:
                area[j] = index[int(a)]
        miss = np.flatnonzero(area < 0)
        n_look += rr.size
        n_fb += miss.size
        if miss.size:
            _, near = tree.query(np.column_stack([xs[miss], ys[miss]]), k=1)
            area[miss] = near
        for k in range(2):
            dk = fields[k][area]
            dk = np.where(np.isfinite(dk), dk, MISSING)
            # per row: min over living CTs (slots 0-4) and living Ts (slots 5-9)
            for side, sel in enumerate([ss < 5, ss >= 5]):
                best = np.full(e - c, MISSING)
                np.minimum.at(best, rr[sel], dk[sel])
                out[c:e, 2 * k + side] = np.minimum(np.where(best < MISSING, best, 10000.0), 10000.0) / 1000.0
        print(f"  {m} rows {e:,} of {N:,}; find_area misses so far {n_fb / max(n_look, 1):.3f} ({time.time() - t0:.0f} s)", flush=True)
    if cap > 0:
        suffix = "_ext_navwalk_test.npy"
    elif os.environ.get("NAVWALK_TRAIN_CENTRES", "0") == "1":
        suffix = "_ext_navwalk_trainsites.npy"
    else:
        suffix = "_ext_navwalk.npy"
    out_path = os.path.join(DATA, f"{m}{suffix}")
    np.save(out_path, out)
    print(f"{m}: wrote {out_path}; shape {out.shape}; no-area fallback share {n_fb / max(n_look, 1):.4f}; "
          f"capped share {(out >= 10.0).mean():.4f} ({time.time() - t0:.0f} s)", flush=True)
    return 0


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: build_nav_walk_allmaps.py <map> [<map> ...]")
    for m in sys.argv[1:]:
        build(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
