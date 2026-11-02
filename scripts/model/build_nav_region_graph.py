"""Map graph from the navigation mesh, for the THGNN-style reimplementation (de_mirage, extended round-state file).

The data-driven graph (build_map_graph.py) learns 48 k-means nodes and edges from TRAIN player positions. This builds the same kind of
graph with no player data at all, so it cannot see the test split:
  nodes      48 regions: k-means over the navigation areas' centroids (x, y, scaled with the round-state window), weighted by area size
  edges      two regions are adjacent when a navigation connection joins an area in one to an area in the other (self loops included)
  assign     each alive player slot to the region of its nearest navigation-area centroid (2D, scaled); -1 for dead or missing slots
Same format as <map>_mapgraph.npz: centroids (K, 2), adjacency (K, K) bool, assign (N, 10) int16.

Writes data/round_state/de_mirage_ext_navgraph.npz.
Run in the cs2gpu environment (numpy, scipy, sklearn):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/build_nav_region_graph.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

ROOT = "G:/pureskill/csds-cs2"
DATA = os.path.join(ROOT, "data", "round_state")
K = 48


def weighted_kmeans(pts, w, k, seed=0, iters=200):
    """Lloyd's k-means with k-means++ seeding and weights, in numpy (sklearn's KMeans crashes natively in this environment)."""
    rng = np.random.default_rng(seed)
    centers = [pts[rng.choice(len(pts), p=w / w.sum())]]
    d2 = ((pts - centers[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        p = w * d2
        centers.append(pts[rng.choice(len(pts), p=p / p.sum())])
        d2 = np.minimum(d2, ((pts - centers[-1]) ** 2).sum(axis=1))
    c = np.array(centers)
    lab = np.zeros(len(pts), dtype=np.int64)
    for _ in range(iters):
        dist = ((pts[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        new = dist.argmin(axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = np.average(pts[m], axis=0, weights=w[m])
    return lab, c


def main():
    t0 = time.time()
    g = np.load(os.path.join(ROOT, "data", "geometry", "de_mirage_nav_graph.npz"))
    cen = g["centroid"][:, :2]
    size = g["size"]
    edges = g["edges"]
    bounds = json.load(open(os.path.join(DATA, "de_mirage_ext.json"), encoding="utf-8"))["window"]
    xy_min = bounds["xy_min"]
    xy_span = bounds["xy_max"] - bounds["xy_min"]
    cen_s = (cen - xy_min) / xy_span
    lab, centers = weighted_kmeans(cen_s, np.maximum(size, 1.0), K, seed=0)
    centroids = centers.astype(np.float32)
    adj = np.zeros((K, K), dtype=bool)
    for a, b in edges:
        r, s = lab[a], lab[b]
        if r != s:
            adj[r, s] = adj[s, r] = True
    np.fill_diagonal(adj, True)
    print(f"regions {K}; mesh edges {len(edges)}; region edges (excl. self) {int(adj.sum() - K) // 2}; "
          f"isolated regions {int(((adj.sum(axis=1) - 1) == 0).sum())} ({time.time() - t0:.0f} s)", flush=True)

    tree = cKDTree(cen_s)
    z = np.load(os.path.join(DATA, "de_mirage_ext.npz"))
    X = z["X"]
    N = X.shape[0]
    assign = np.full((N, 10), -1, dtype=np.int16)
    chunk = 500_000
    for s in range(0, N, chunk):
        block = X[s:s + chunk]
        alive = block[:, :, 0] > 0.5
        xy = block[:, :, 1:3][alive].astype(np.float64)
        _, nearest = tree.query(xy, k=1)
        out = np.full(alive.shape, -1, dtype=np.int16)
        out[alive] = lab[nearest].astype(np.int16)
        assign[s:s + chunk] = out
        print(f"  rows {s + block.shape[0]:,} of {N:,} ({time.time() - t0:.0f} s)", flush=True)
    print(f"assigned alive slots ({time.time() - t0:.0f} s)", flush=True)

    out = os.path.join(DATA, "de_mirage_ext_navgraph.npz")
    np.savez(out, centroids=centroids, adjacency=adj, assign=assign)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
