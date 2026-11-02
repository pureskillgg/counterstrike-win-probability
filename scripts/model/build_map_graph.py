"""Data-driven map graph for the THGNN-style reimplementation (de_mirage).

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\model\\build_map_graph.py --map de_mirage

The paper hand-builds the map graph with a former professional player. Here the nodes are k-means centroids of the TRAIN
player positions (x, y, scaled as in round_state), and an edge joins two nodes when a player moved between them in consecutive
seconds of a train round (at least MIN_EDGE_COUNT transitions, made symmetric). The test split is never used.
Writes data/round_state/<map>_mapgraph.npz: centroids (K, 2), adjacency (K, K) bool with self loops, assign (N, 10) int16
(cluster of each slot, -1 for dead or missing slots).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "round_state"
K = 48
SAMPLE = 400_000
MIN_EDGE_COUNT = 5
TRAIN = 0


def next_second_rows(match_idx, round_id, second):
    """Row index of the same match and round one second later, or -1."""
    key = ((match_idx.astype(np.int64) * 128 + round_id.astype(np.int64)) * 1024) + second.astype(np.int64)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    target = key + 1
    pos = np.searchsorted(skey, target)
    pos_c = np.minimum(pos, skey.size - 1)
    found = (pos < skey.size) & (skey[pos_c] == target)
    return np.where(found, order[pos_c], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="de_mirage")
    ap.add_argument("--stem", default=None, help="round-state file stem (default: the map name), e.g. de_mirage_ext")
    a = ap.parse_args()
    map_name = a.map
    stem = a.stem or map_name
    arr = np.load(DATA / f"{stem}.npz")
    X = arr["X"]
    split = arr["split"]
    N = X.shape[0]
    alive = X[:, :, 0] > 0.5
    xy = X[:, :, 1:3]

    rng = np.random.default_rng(0)
    train_pts = xy[(split == TRAIN)[:, None] & alive]
    sample = train_pts[rng.choice(len(train_pts), size=min(SAMPLE, len(train_pts)), replace=False)]
    km = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=20_000, n_init=3).fit(sample)
    centroids = km.cluster_centers_.astype(np.float32)
    print(f"fitted {K} centroids on {sample.shape[0]} train positions", flush=True)

    assign = np.full((N, 10), -1, dtype=np.int16)
    chunk = 500_000
    for s in range(0, N, chunk):
        a = alive[s:s + chunk]
        flat = xy[s:s + chunk][a]
        lab = km.predict(flat).astype(np.int16)
        block = np.full(a.shape, -1, dtype=np.int16)
        block[a] = lab
        assign[s:s + chunk] = block
    print("assigned every alive slot to a node", flush=True)

    nxt = next_second_rows(arr["match_idx"], arr["round"], arr["second"])
    has_next = nxt >= 0
    trans = np.zeros((K, K), dtype=np.int64)
    rows = np.flatnonzero(has_next & (split == TRAIN))
    for slot in range(10):
        a_i = assign[rows, slot]
        a_j = assign[nxt[rows], slot]
        ok = (a_i >= 0) & (a_j >= 0) & (a_i != a_j)
        np.add.at(trans, (a_i[ok].astype(np.int64), a_j[ok].astype(np.int64)), 1)
    sym = trans + trans.T
    adj = sym >= MIN_EDGE_COUNT
    np.fill_diagonal(adj, True)
    print(f"edges (undirected, excluding self loops): {int(adj.sum() - K) // 2}; isolated nodes: "
          f"{int(((adj.sum(axis=1) - 1) == 0).sum())}", flush=True)

    out = DATA / f"{stem}_mapgraph.npz"
    np.savez(out, centroids=centroids, adjacency=adj, assign=assign)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
