"""M4 sites: are site codes stable per map, can planter positions recover A and B per map, can the
clusters be named from place names, and is the geometry the same in 2024?

Inputs (scratchpad, from M4_build.py): plants_all_2026.parquet, rounds_2026.parquet.
Also reads: data\\state_sample\\place_names.parquet (1 Hz sample, 150 matches) and the 2024 tome
(header and bomb_state planted rows, and round_end for the round-13 check). Read-only.
All clustering here is descriptive and uses all 2026 kept planted rounds. The model test
(M4_models.py) refits clusters on training matches only.
"""
import glob
import os
import collections
import math

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SCR = (r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
       r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M4")
PN = r"G:\pureskill\csds-cs2\data\state_sample\place_names.parquet"
T2024 = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF24 = ".2023-12-28,2024-02-25"
SEED = 0


def read24(channel, cols=None):
    folder = T2024 if channel == "header" else os.path.join(T2024, f"basic_{channel}{SUF24}")
    if channel == "header":
        folder = os.path.join(T2024, "header")
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def sep_ratio(X, lab, centers):
    """centre distance / mean within-cluster RMS distance (2 clusters)."""
    d = np.linalg.norm(centers[0] - centers[1])
    rms = []
    for j in (0, 1):
        pts = X[lab == j]
        if len(pts) == 0:
            return float("nan")
        rms.append(np.sqrt(((pts - centers[j]) ** 2).sum(1).mean()))
    return d / np.mean(rms)


def main():
    P = pl.read_parquet(os.path.join(SCR, "plants_all_2026.parquet"))
    R = pl.read_parquet(os.path.join(SCR, "rounds_2026.parquet"))
    print("== 1. Site codes per map (all 2026 header matches with planted rows)")
    print("plant rows:", P.height, "null site_code:", int(P["site_code"].is_null().sum()))
    per_match = (P.group_by(["match_key", "map_name"])
                 .agg(pl.col("site_code").n_unique().alias("n_codes"),
                      pl.col("site_code").len().alias("n_plants")))
    codes_by_map = {}
    for mp in sorted(P["map_name"].unique().to_list()):
        sub = P.filter(pl.col("map_name") == mp)
        pm = per_match.filter(pl.col("map_name") == mp)
        mc = sub.group_by("site_code").agg(pl.len().alias("n"),
                                           pl.col("match_key").n_unique().alias("n_matches"))
        recur = mc.filter(pl.col("n_matches") > 1).height
        codes_by_map[mp] = mc
        dist = pm["n_codes"].value_counts().sort("n_codes")
        dist_s = ", ".join(f"{r['n_codes']}:{r['count']}" for r in dist.iter_rows(named=True))
        top = mc.sort("n_matches", descending=True).head(4)
        top_s = "; ".join(f"{r['site_code']}({r['n_matches']} matches)" for r in top.iter_rows(named=True))
        print(f"  {mp}: matches with plants={pm.height}; distinct codes={mc.height}; codes in >1 match="
              f"{recur}; distinct codes per match (count of matches) {{{dist_s}}}; top codes: {top_s}")

    print("\n== 2. Planter-position clusters per map (k=2, 3D, descriptive, all 2026 kept planted rounds)")
    K = P.filter(pl.col("kept") & pl.col("x").is_not_null() & pl.col("y").is_not_null()
                 & pl.col("z").is_not_null())
    K = K.unique(subset=["match_key", "round"], keep="first")
    lab_frames = []
    centers_all = {}
    for mp in sorted(K["map_name"].unique().to_list()):
        sub = K.filter(pl.col("map_name") == mp)
        X = sub.select(["x", "y", "z"]).to_numpy().astype(np.float64)
        if len(X) < 10:
            print(f"  {mp}: n={len(X)} too few")
            continue
        km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(X)
        lab = km.labels_
        sizes = np.bincount(lab, minlength=2)
        big = int(np.argmax(sizes))
        # relabel: 0 = larger cluster (A, descriptive), 1 = smaller
        lab2 = np.where(lab == big, 0, 1)
        c = np.vstack([km.cluster_centers_[big], km.cluster_centers_[1 - big]])
        samp = np.random.default_rng(SEED).choice(len(X), size=min(4000, len(X)), replace=False)
        sil = silhouette_score(X[samp], lab2[samp]) if len(set(lab2[samp])) == 2 else float("nan")
        sr = sep_ratio(X, lab2, c)
        centers_all[mp] = c
        lab_frames.append(sub.select(["match_key", "round", "map_name", "site_code"])
                          .with_columns(pl.Series("cluster", lab2)))
        print(f"  {mp}: n={len(X)} sizes(cluster0,cluster1)=({int(sizes[big])},{int(sizes[1 - big])}) "
              f"centres0={np.round(c[0]).astype(int).tolist()} centres1={np.round(c[1]).astype(int).tolist()} "
              f"centre_dist={np.linalg.norm(c[0]-c[1]):.0f} sep_ratio={sr:.2f} silhouette={sil:.3f}")
    CL = pl.concat(lab_frames)
    CL.write_parquet(os.path.join(SCR, "planter_clusters_2026_descriptive.parquet"))

    print("\n== 3. Site code vs cluster (purity of each code inside its map)")
    for mp in sorted(centers_all):
        sub = CL.filter(pl.col("map_name") == mp)
        ct = sub.group_by(["site_code", "cluster"]).agg(pl.len().alias("n"))
        tot = sub.group_by("site_code").agg(pl.len().alias("tot"))
        j = ct.join(tot, on="site_code")
        pure = (j.group_by("site_code").agg((pl.col("n").max() / pl.col("tot").first()).alias("purity"),
                                            pl.col("tot").first().alias("tot"))
                .filter(pl.col("tot") >= 5))
        pv = pure["purity"].to_numpy() if pure.height else np.array([np.nan])
        n_codes_ge5 = pure.height
        share_pure95 = float(np.mean(pv >= 0.95)) if pure.height else float("nan")
        # within-match: matches with 2 distinct codes, do the two codes take different clusters?
        mm = (sub.group_by(["match_key", "site_code"]).agg(pl.col("cluster").mode().first().alias("cl"))
              .group_by("match_key").agg(pl.col("site_code").n_unique().alias("nc"),
                                         pl.col("cl").n_unique().alias("ncl")))
        two = mm.filter(pl.col("nc") == 2)
        diff = int((two["ncl"] == 2).sum()) if two.height else 0
        print(f"  {mp}: codes with >=5 planted rounds={n_codes_ge5}; share of those codes with purity>=0.95: "
              f"{share_pure95:.2f}; median purity={np.median(pv):.3f}; matches with 2 codes={two.height}, "
              f"of which codes go to different clusters={diff}")

    print("\n== 4. Name the clusters with place names (1 Hz sample, 150 matches)")
    pn = pl.read_parquet(PN, columns=["match_key", "round", "tick", "player_id", "place_name",
                                      "is_in_bomb_zone"])
    print("place_name values in bomb zone (count):",
          pn.filter(pl.col("is_in_bomb_zone")).group_by("place_name").len().sort("len", descending=True)
          .head(12).iter_rows().__iter__().__class__.__name__)
    bz = pn.filter(pl.col("is_in_bomb_zone"))
    print(bz.group_by("place_name").len().sort("len", descending=True).head(12))
    sample_keys = set(pn["match_key"].unique().to_list())
    # plant rows in the sample: nearest-tick bomb-zone place name within 1 s
    pl_s = CL.filter(pl.col("match_key").is_in(list(sample_keys)))
    Ps = (P.filter(pl.col("kept") & pl.col("match_key").is_in(list(sample_keys)))
          .unique(subset=["match_key", "round"], keep="first")
          .select(["match_key", "round", "tick", "map_name", "x", "y", "z"]))
    Ps = Ps.join(pl_s.select(["match_key", "round", "cluster"]), on=["match_key", "round"], how="inner")
    names = []
    bzg = bz.group_by(["match_key", "round"]).agg(pl.col("tick"), pl.col("place_name"))
    bzd = {(r["match_key"], r["round"]): (r["tick"], r["place_name"]) for r in bzg.iter_rows(named=True)}
    for r in Ps.iter_rows(named=True):
        key = (r["match_key"], r["round"])
        if key not in bzd:
            names.append(None)
            continue
        ticks, places = bzd[key]
        near = [p for t, p in zip(ticks, places) if abs(t - r["tick"]) <= 64]
        if not near:
            names.append(None)
            continue
        names.append(collections.Counter(near).most_common(1)[0][0])
    Ps = Ps.with_columns(pl.Series("place_at_plant", names))
    ct = Ps.filter(pl.col("place_at_plant").is_not_null()).group_by(["map_name", "cluster", "place_at_plant"]).len()
    print("planted rounds in sample:", Ps.height, "with a bomb-zone place name:",
          int(Ps["place_at_plant"].is_not_null().sum()))
    print(ct.sort(["map_name", "cluster", "len"], descending=[False, False, True]))

    print("\n== 5. Post-plant outcome by cluster (2026 kept rounds, descriptive, unadjusted)")
    Rp = R.filter(pl.col("plant_tick").is_not_null() & pl.col("plant_x").is_not_null())
    Rp = Rp.with_columns(pl.col("round").cast(pl.Int64))
    Cj = CL.select(["match_key", "round", "cluster"]).with_columns(pl.col("round").cast(pl.Int64))
    Rp = Rp.join(Cj, on=["match_key", "round"], how="inner")
    Rp = Rp.with_columns((pl.col("win_reason_code") == 1).cast(pl.Int8).alias("exploded"))
    out_rows = []
    for (mp, cl), g in Rp.group_by(["map_name", "cluster"]):
        n = g.height
        k = int(g["ct_win"].sum())
        lo, hi = wilson(k, n)
        ex = int(g["exploded"].sum())
        out_rows.append((mp, int(cl), n, k / n, lo, hi, ex / n))
    print("map, cluster, planted rounds, CT win, Wilson 95%, exploded share")
    for r in sorted(out_rows):
        print(f"  {r[0]} c{r[1]} n={r[2]} CT={r[3]:.3f} [{r[4]:.3f}, {r[5]:.3f}] exploded={r[6]:.3f}")
    # 2x? plants share by cluster
    print("\nplant counts by map x cluster and batch (1008 = match_date 2026-10-08):")
    Rp2 = Rp.with_columns(pl.col("batch_1008").cast(pl.Int8))
    print(Rp2.group_by(["map_name", "cluster", "batch_1008"]).agg(pl.len().alias("n"),
                                                             pl.col("ct_win").mean().alias("ct_win")).sort(["map_name", "cluster", "batch_1008"]))

    print("\n== 6. 2024 tome: cluster geometry and site codes (cross-era check)")
    h24 = read24("header", ["key", "map_name", "build_num", "match_date", "max_rounds"])
    b24 = read24("bomb_state", ["match_key", "round", "tick", "event_type", "site_code",
                                "player_team_code", "player_x_pos", "player_y_pos", "player_z_pos"])
    pl24 = b24.filter(pl.col("event_type") == "bomb_planted").join(
        h24.select(["key", "map_name"]).rename({"key": "match_key"}), on="match_key", how="left")
    print("2024 planted rows:", pl24.height, "matches:", pl24["match_key"].n_unique(),
          "null site_code:", int(pl24["site_code"].is_null().sum()))
    pm24 = pl24.group_by(["match_key", "map_name"]).agg(pl.col("site_code").n_unique().alias("nc"))
    for mp in sorted(centers_all):
        sub = pl24.filter(pl.col("map_name") == mp)
        if sub.height == 0:
            print(f"  {mp}: no 2024 planted rows")
            continue
        X = sub.select(["player_x_pos", "player_y_pos", "player_z_pos"]).to_numpy().astype(np.float64)
        c = centers_all[mp]
        d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2)
        nearest = d.argmin(1)
        dmin = d.min(1)
        share_1000 = float(np.mean(dmin <= 1000))
        nc_m = pm24.filter(pl.col("map_name") == mp)["nc"]
        print(f"  {mp}: 2024 planted n={len(X)}; within 1000 units of a 2026 centre: {share_1000:.3f}; "
              f"median distance to nearest centre={np.median(dmin):.0f}; share nearest c0={np.mean(nearest==0):.3f}; "
              f"2024 distinct codes per match: {nc_m.value_counts().sort('nc').iter_rows().__class__.__name__}")
    print("  2024 codes per match (all maps):", pm24["nc"].value_counts().sort("nc").to_dict(as_series=False))

    print("\n== 7. Round-13 bomb-side mismatch by map and build (2024; win_reason 1 = exploded, 7 = defused)")
    re24 = read24("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                "player_count"])
    r13 = (re24.filter((pl.col("round") == 13) & (pl.col("player_count") == 20)
                       & pl.col("win_reason_code").is_in([1, 7]))
           .join(h24.select(["key", "map_name", "build_num"]).rename({"key": "match_key"}),
                 on="match_key", how="left"))
    r13 = r13.with_columns(
        (((pl.col("win_reason_code") == 1) & (pl.col("winner_team_code") != 2))
         | ((pl.col("win_reason_code") == 7) & (pl.col("winner_team_code") != 3))).alias("mismatch"))
    print("round 13 rows with a bomb outcome:", r13.height, "mismatches:", int(r13["mismatch"].sum()))
    tab = (r13.group_by(["build_num", "map_name"]).agg(pl.len().alias("n"),
                                                       pl.col("mismatch").sum().alias("mm"))
           .sort(["build_num", "map_name"]))
    print(tab)


if __name__ == "__main__":
    main()
