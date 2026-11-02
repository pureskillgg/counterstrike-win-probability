"""Export the de_mirage navigation mesh as a graph: area centroids, area sizes, and the connections between areas.

Run in the navigation environment (awpy 3.x, which exposes area() with centroid, size and connections):
  C:/Users/13144/pureskill-nav312/Scripts/python.exe scripts/model/export_nav_graph.py
Writes data/geometry/de_mirage_nav_graph.npz: area_id (A,), centroid (A, 3), size (A,), edges (E, 2) as indices into area_id.
Each undirected connection appears once (lower index first). Used by build_nav_region_graph.py.
"""
import os

import numpy as np
import awpy._awpy as core

ROOT = "G:/pureskill/csds-cs2"
OUT = os.path.join(ROOT, "data", "geometry", "de_mirage_nav_graph.npz")


def main():
    nav = core.NavMesh(os.path.join(ROOT, "data", "geometry", "navs", "de_mirage.nav"))
    ids = sorted(int(a) for a in nav.areas["area_id"].to_list())
    index = {a: i for i, a in enumerate(ids)}
    centroid = np.zeros((len(ids), 3))
    size = np.zeros(len(ids))
    edges = set()
    for a in ids:
        d = nav.area(a)
        i = index[a]
        centroid[i] = d["centroid"]
        size[i] = float(d["size"])
        for b in d["connections"]:
            b = int(b)
            if b in index and b != a:
                j = index[b]
                edges.add((min(i, j), max(i, j)))
    edges = np.array(sorted(edges), dtype=np.int64)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, area_id=np.array(ids, dtype=np.int64), centroid=centroid, size=size, edges=edges)
    print(f"areas {len(ids)}, undirected edges {len(edges)}, wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
