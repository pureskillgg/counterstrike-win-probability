"""C4 site-geometry check (read-only, CPU). Per map: how well the two bomb-site planter clusters are separated, fitted on TRAIN
planted rounds only, with the validation planters used only as an out-of-sample check.

Why: the distance features (train_temporal_allmaps.straight_distances, sitecheck_geo.py) are distances to two site centres. If the
two centres are close together, the two distances carry little site-specific information. Earlier numbers (notes/eda/
ablation_positions_map_control.md, 150-match sample, 3D) put de_nuke at a 544-unit centre distance with silhouette 0.54; the other
maps sat at 2,300 to 3,100 with silhouette 0.92 to 0.95. This script repeats the check on the current training split, in 2D, which is
the plane the distance features use. The plant file carries no height, so the check is 2D only.

Also reported: for each map, the mean paired log-loss gain of the distance tree (SN minus A, validation seconds, mean over seeds 0-2;
negative = distances help), and the rank correlation across the seven maps between centre distance and that gain.

Writes data/diagnostics/c4_site_geometry/c4_site_geometry.json and prints one table.
Run: G:/pureskill/envs/tabicl/Scripts/python.exe -I -X utf8 -u scripts/model/c4_site_geometry.py
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
PLANTS = ROOT / "data" / "positions_1hz" / "_plants"
PREDS = ROOT / "data" / "diagnostics" / "preds"
OUT = ROOT / "data" / "diagnostics" / "c4_site_geometry"
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEYS = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
SITE_RADIUS = 1000.0


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def spearman(a, b):
    ra, rb = rank(np.asarray(a)), rank(np.asarray(b))
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def planter_table():
    man = json.load(open(ROOT / "data" / "combined" / "split_manifest.json", encoding="utf-8"))["assign"]
    P = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(PLANTS / "*.parquet")))], how="vertical_relaxed")
    M = pl.read_parquet(ROOT / "data" / "combined" / "matches.parquet").select("match_key", "map_name")
    P = P.join(M, on="match_key").drop_nulls(["x", "y"])
    P = P.with_columns(pl.col("match_key").replace_strict(man, default="none").alias("split"))
    return P


def fit_map(g_train, g_val):
    xy = g_train.select(["x", "y"]).to_numpy().astype(np.float64)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(xy)
    lab = km.labels_
    c = km.cluster_centers_
    sizes = [int((lab == k).sum()) for k in (0, 1)]
    big = 0 if sizes[0] >= sizes[1] else 1  # site A = the larger cluster, as in the earlier note
    order = [big, 1 - big]
    c = c[order]
    lab = np.where(lab == order[0], 0, 1)
    sizes = [int((lab == k).sum()) for k in (0, 1)]
    rms = [float(np.sqrt(np.mean(((xy[lab == k] - c[k]) ** 2).sum(1)))) for k in (0, 1)]
    D = float(np.linalg.norm(c[0] - c[1]))
    sil = float(silhouette_score(xy, lab))
    res = {
        "train_plants": int(xy.shape[0]), "size_A": sizes[0], "size_B": sizes[1],
        "centre_A": [round(float(v), 1) for v in c[0]], "centre_B": [round(float(v), 1) for v in c[1]],
        "centre_distance_2d": round(D, 1), "rms_A": round(rms[0], 1), "rms_B": round(rms[1], 1),
        "separation_ratio_2d": round(D / float(np.mean(rms)), 2), "silhouette_2d": round(sil, 3),
        "val_plants": int(g_val.height),
    }
    if g_val.height:
        vxy = g_val.select(["x", "y"]).to_numpy().astype(np.float64)
        dmin = np.sqrt(((vxy[:, None, :] - c[None]) ** 2).sum(-1)).min(1)
        res["val_share_within_1000"] = round(float(np.mean(dmin <= SITE_RADIUS)), 4)
        res["val_median_dist"] = round(float(np.median(dmin)), 1)
    return res


def distance_gain():
    """Per map: mean over seeds of the paired validation log-loss difference, SN minus A (negative = SN better)."""
    per = {m: [] for m in MAPS}
    for s in SEEDS:
        sn = pl.read_parquet(PREDS / f"allmaps_xgb_SN_seed{s}.parquet")
        sn = sn.filter(pl.col("split") == "val").select(KEYS + ["map_name", "label", "p"]).rename({"p": "sn"})
        a = pl.read_parquet(PREDS / f"allmaps_xgb_A_seed{s}.parquet")
        a = a.filter(pl.col("split") == "val").select(KEYS + ["p"]).rename({"p": "a"})
        f = sn.join(a, on=KEYS, how="inner")
        y = f["label"].to_numpy().astype(np.float64)
        d = ll(y, f["sn"].to_numpy()) - ll(y, f["a"].to_numpy())
        mp = f["map_name"].to_numpy()
        for m in MAPS:
            sel = mp == m
            per[m].append(float(d[sel].mean()))
    return {m: float(np.mean(v)) for m, v in per.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = planter_table()
    print(f"plants with positions and a map: {P.height:,}", flush=True)
    maps = {}
    for m in MAPS:
        g = P.filter(pl.col("map_name") == m)
        maps[m] = fit_map(g.filter(pl.col("split") == "train"), g.filter(pl.col("split") == "val"))
        print(f"{m}: {maps[m]}", flush=True)

    gain = distance_gain()
    for m in MAPS:
        maps[m]["sn_minus_a_logloss"] = round(gain[m], 5)
    dist = [maps[m]["centre_distance_2d"] for m in MAPS]
    summary = {
        "spearman_centre_distance_vs_sn_minus_a": round(spearman(dist, [gain[m] for m in MAPS]), 3),
        "spearman_separation_ratio_vs_sn_minus_a": round(
            spearman([maps[m]["separation_ratio_2d"] for m in MAPS], [gain[m] for m in MAPS]), 3),
        "spearman_centre_distance_vs_sn_minus_a_without_nuke": round(
            spearman([maps[m]["centre_distance_2d"] for m in MAPS if m != "de_nuke"],
                     [gain[m] for m in MAPS if m != "de_nuke"]), 3),
    }
    out = {"maps": maps, "summary": summary, "note": "2D k-means k=2 on train planters; validation planters out of sample; "
                                                    "sn_minus_a_logloss is validation, mean over seeds 0-2, negative = distances help"}
    with open(OUT / "c4_site_geometry.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print("\nmap         train  A/B          centre_2d  ratio  sil    val<=1000  SN-A logloss")
    for m in MAPS:
        r = maps[m]
        print(f"{m:11s} {r['train_plants']:5d}  {r['size_A']}/{r['size_B']:<8d}  {r['centre_distance_2d']:9.1f}  "
              f"{r['separation_ratio_2d']:5.2f}  {r['silhouette_2d']:.3f}  {r.get('val_share_within_1000', float('nan')):9.4f}  "
              f"{r['sn_minus_a_logloss']:+.5f}")
    print("\nsummary:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
