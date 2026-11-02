"""Travel-time map control per snapshot (de_mirage 2026, every 10th second, same rows as map_control_features.py), for T7.

For each snapshot, every living player's position is mapped to a navigation area (awpy find_area). Each side's control time for an area
is the minimum over that side's living players of (centroid-graph path distance from the player's area to the area) divided by the
player's held-weapon speed. The side that reaches an area first controls it; ties and areas neither side can reach count as neutral.
Features are the size-weighted control shares: t7_ct, t7_t, and t7_net = t7_ct - t7_t.

Speed (units per second) by inventory code, from the published table recorded in the T6 registration (tradeit.gg, community-compiled).
Primary codes; a player with no primary (inv_primary 0) uses the secondary code. A player whose code is not in the table is excluded and
counted. Knife, scope and crouch state are not in the stored data.

Writes data/diagnostics/map_control_time_de_mirage_stride10.parquet (match_key, round, second, t7_ct, t7_t, t7_net) and prints the
exclusion counts. Snapshots with no evaluable CT or T player are left null.
Run in the navigation environment (awpy 3.x, scipy):
  C:/Users/<user>/pureskill-nav312/Scripts/python.exe scripts/model/map_control_time.py
"""
import glob
import os
import sys
import time

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import awpy._awpy as core

ROOT = "G:/pureskill/csds-cs2"
DIAG = os.path.join(ROOT, "data", "diagnostics")
ROWS = os.path.join(DIAG, "map_control_de_mirage_stride10.parquet")
GRAPH = os.path.join(ROOT, "data", "geometry", "de_mirage_nav_graph.npz")
NAV = os.path.join(ROOT, "data", "geometry", "navs", "de_mirage.nav")
OUT = os.path.join(DIAG, "map_control_time_de_mirage_stride10.parquet")
KEY = ["match_key", "round", "second"]
CT, T = 3, 2

# published maximum movement speed, units per second (tradeit.gg), by inventory code
PRIMARY = {7: 215, 60: 225, 16: 225, 9: 200, 13: 215, 40: 230, 33: 220, 34: 240, 17: 240, 10: 220, 25: 215, 23: 235,
           39: 210, 8: 220, 19: 230, 35: 220, 27: 225, 28: 150, 24: 230, 26: 240, 29: 210, 38: 215, 11: 215, 14: 195}
SECONDARY = {4: 240, 61: 240, 1: 230, 32: 240, 30: 240, 3: 240, 36: 240, 2: 240, 63: 240, 64: 180}


def speed_lut(table):
    lut = np.full(256, np.nan)
    for code, v in table.items():
        lut[code] = v
    return lut


PRIM_LUT, SEC_LUT = speed_lut(PRIMARY), speed_lut(SECONDARY)


def player_speed(prim, sec):
    p = PRIM_LUT[prim]
    s = SEC_LUT[sec]
    return np.where(~np.isnan(p), p, np.where(prim == 0, s, np.nan))


def main():
    t0 = time.time()
    nav = core.NavMesh(NAV)
    g = np.load(GRAPH)
    area_id, centroid, size, edges = g["area_id"], g["centroid"], g["size"], g["edges"]
    n = len(area_id)
    w = np.linalg.norm(centroid[edges[:, 0]] - centroid[edges[:, 1]], axis=1)
    A = coo_matrix((np.r_[w, w], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    D = dijkstra(A, directed=False)  # n x n shortest path distances, game units
    idx_of = np.full(int(area_id.max()) + 2, -1, dtype=np.int64)
    idx_of[area_id] = np.arange(n)
    print(f"graph: {n} areas, {len(edges)} edges, distance matrix {D.shape} ({time.time() - t0:.0f} s)", flush=True)

    rows = pl.read_parquet(ROWS, columns=KEY)
    keys = set(map(tuple, rows.select(KEY).rows()))
    matches = sorted(set(rows["match_key"].to_list()))
    out = {k: [] for k in KEY + ["t7_ct", "t7_t", "t7_net"]}
    stats = {"snapshots": 0, "no_side": 0, "players_no_speed": 0, "players_no_area": 0, "players_used": 0}

    for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "2026-*.parquet"))):
        day = os.path.basename(f)
        pos = pl.read_parquet(f, columns=KEY + ["player_id", "side", "x", "y", "z"]).filter(
            pl.col("match_key").is_in(matches) & (pl.col("second") % 10 == 0) & pl.col("side").is_in([CT, T]))
        if pos.height == 0:
            continue
        st = pl.read_parquet(os.path.join(ROOT, "data", "status_1hz", day),
                             columns=KEY + ["player_id", "health", "inv_primary", "inv_secondary"])
        j = pos.join(st, on=KEY + ["player_id"], how="inner").filter(pl.col("health") > 0)
        j = j.with_columns(pl.col("inv_primary").cast(pl.Int64), pl.col("inv_secondary").cast(pl.Int64),
                           pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
        pr = j["inv_primary"].to_numpy()
        sc = j["inv_secondary"].to_numpy()
        spd = player_speed(np.clip(pr, 0, 255), np.clip(sc, 0, 255))
        j = j.with_columns(pl.Series("speed", spd))
        grp = j.partition_by(KEY, as_dict=True, maintain_order=True)
        for kk, df in grp.items():
            if (tuple(kk[:3]) if isinstance(kk, tuple) else kk) not in keys:
                continue
            stats["snapshots"] += 1
            xs, ys, zs = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
            sides, spds = df["side"].to_numpy(), df["speed"].to_numpy()
            tcts, tts = [], []
            for x, y, z, s, v in zip(xs, ys, zs, sides, spds):
                if np.isnan(v):
                    stats["players_no_speed"] += 1
                    continue
                a = nav.find_area((float(x), float(y), float(z)))
                ia = idx_of[a] if (a is not None and 0 <= a < len(idx_of)) else -1
                if ia < 0:
                    stats["players_no_area"] += 1
                    continue
                stats["players_used"] += 1
                (tcts if s == CT else tts).append(D[ia] / v)
            if not tcts or not tts:
                stats["no_side"] += 1
                for k in ["t7_ct", "t7_t", "t7_net"]:
                    out[k].append(None)
                for k, v in zip(KEY, kk if isinstance(kk, tuple) else (kk,)):
                    out[k].append(v)
                continue
            tc = np.min(np.vstack(tcts), axis=0)
            tt = np.min(np.vstack(tts), axis=0)
            ct_win = (tc < tt) & np.isfinite(tc)
            t_win = (tt < tc) & np.isfinite(tt)
            total = size.sum()
            ct_frac = float(size[ct_win].sum() / total)
            t_frac = float(size[t_win].sum() / total)
            for k, v in zip(KEY, kk if isinstance(kk, tuple) else (kk,)):
                out[k].append(v)
            out["t7_ct"].append(ct_frac)
            out["t7_t"].append(t_frac)
            out["t7_net"].append(ct_frac - t_frac)
        print(f"  {day}: {stats['snapshots']:,} snapshots so far ({time.time() - t0:.0f} s)", flush=True)

    R = pl.DataFrame(out).with_columns([pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)]).sort(KEY)
    R.write_parquet(OUT)
    print(json_like(stats), flush=True)
    print(f"wrote {OUT}: {R.height:,} snapshots ({time.time() - t0:.0f} s)", flush=True)
    return 0


def json_like(d):
    return "stats " + ", ".join(f"{k} {v:,}" for k, v in d.items())


if __name__ == "__main__":
    sys.exit(main())
