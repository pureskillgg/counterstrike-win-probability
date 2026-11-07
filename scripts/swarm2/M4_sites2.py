"""M4 sites follow-up: full place-name crosstabs, code-to-place agreement, codes shared across maps,
code pairs per map, and the 2024 cross-era geometry check (fixed header columns).

Reads: scratchpad plants_all_2026.parquet and planter_clusters_2026_descriptive.parquet (from
M4_sites.py), data\\state_sample\\place_names.parquet, and the 2024 tome (read-only).
"""
import collections
import glob
import os

import numpy as np
import polars as pl

SCR = (r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
       r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M4")
PN = r"G:\pureskill\csds-cs2\data\state_sample\place_names.parquet"
T2024 = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF24 = ".2023-12-28,2024-02-25"


def read24(channel, cols=None):
    folder = os.path.join(T2024, "header") if channel == "header" else os.path.join(
        T2024, f"basic_{channel}{SUF24}")
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def main():
    h24_names = pl.read_parquet(sorted(glob.glob(os.path.join(T2024, "header", "dataframe_*")))[0]).columns
    print("2024 header columns with key/match/round/date/map/build:",
          [c for c in h24_names if any(s in c for s in ("key", "match", "round", "date", "map", "build", "rounds"))])

    P = pl.read_parquet(os.path.join(SCR, "plants_all_2026.parquet"))
    CL = pl.read_parquet(os.path.join(SCR, "planter_clusters_2026_descriptive.parquet"))
    pn = pl.read_parquet(PN, columns=["match_key", "round", "tick", "place_name", "is_in_bomb_zone"])
    bz = pn.filter(pl.col("is_in_bomb_zone"))
    bzd = {(r["match_key"], r["round"]): (r["tick"], r["place_name"])
           for r in bz.group_by(["match_key", "round"]).agg(pl.col("tick"), pl.col("place_name"))
           .iter_rows(named=True)}
    sample = set(pn["match_key"].unique().to_list())

    Ps = (P.filter(pl.col("kept") & pl.col("match_key").is_in(list(sample)))
          .unique(subset=["match_key", "round"], keep="first")
          .select(["match_key", "round", "tick", "map_name", "site_code"]))
    Ps = Ps.join(CL.select(["match_key", "round", "cluster"]), on=["match_key", "round"], how="inner")
    names = []
    for r in Ps.iter_rows(named=True):
        v = bzd.get((r["match_key"], r["round"]))
        if v is None:
            names.append(None)
            continue
        ticks = bz.filter((pl.col("match_key") == r["match_key"]) & (pl.col("round") == r["round"]))
        cand = [p for t, p in zip(ticks["tick"].to_list(), ticks["place_name"].to_list())
                if abs(t - r["tick"]) <= 64]
        names.append(collections.Counter(cand).most_common(1)[0][0] if cand else None)
    Ps = Ps.with_columns(pl.Series("place", names))
    Q = Ps.filter(pl.col("place").is_in(["BombsiteA", "BombsiteB"]))
    print(f"\nSample planted rounds with a named bomb site at plant: {Q.height} of {Ps.height}")

    print("\n== map x cluster x place name (cluster 0 = larger cluster, descriptive)")
    for mp in sorted(Q["map_name"].unique().to_list()):
        g = Q.filter(pl.col("map_name") == mp)
        ct = g.group_by(["cluster", "place"]).len().sort(["cluster", "place"]).iter_rows()
        parts = [f"c{c} {p}={n}" for c, p, n in ct]
        # best mapping cluster -> place, and agreement
        tab = g.group_by(["cluster", "place"]).len()
        agree = 0
        for c in (0, 1):
            sub = tab.filter(pl.col("cluster") == c)
            if sub.height:
                agree += int(sub["len"].max())
        print(f"  {mp}: n={g.height}; " + "; ".join(parts) + f"; cluster-to-place agreement="
              f"{agree / g.height:.3f}")

    print("\n== code x place name in the sample (does a site code name a fixed place within a map?)")
    for mp in sorted(Q["map_name"].unique().to_list()):
        g = Q.filter(pl.col("map_name") == mp)
        ctab = g.group_by(["site_code", "place"]).len()
        tot = g.group_by("site_code").agg(pl.len().alias("tot"))
        j = ctab.join(tot, on="site_code")
        pur = (j.group_by("site_code").agg((pl.col("len").max() / pl.col("tot").first()).alias("p"),
                                           pl.col("tot").first().alias("tot")))
        big = pur.filter(pl.col("tot") >= 3)
        codes_ab = (g.group_by(["site_code", "place"]).len()
                    .pivot(on="place", index="site_code", values="len").fill_null(0))
        codes_ab_s = "; ".join(f"{r['site_code']}: A={r.get('BombsiteA', 0)} B={r.get('BombsiteB', 0)}"
                               for r in codes_ab.sort("site_code").head(8).iter_rows(named=True))
        print(f"  {mp}: codes with >=3 named rounds={big.height}; median purity of code->place="
              f"{float(np.median(big['p'].to_numpy())) if big.height else float('nan'):.3f}; "
              f"first codes: {codes_ab_s}")

    print("\n== codes shared across maps (same numeric code on two or more maps, 2026 all plants)")
    mc = P.group_by(["site_code", "map_name"]).agg(pl.len().alias("n"))
    shared = (mc.group_by("site_code").agg(pl.col("map_name").n_unique().alias("nmaps"),
                                           pl.col("n").sum().alias("plants"))
              .filter(pl.col("nmaps") > 1).sort("plants", descending=True))
    print(f"codes on >1 map: {shared.height} of {mc['site_code'].n_unique()} distinct codes; plant rows on "
          f"shared codes: {int(shared['plants'].sum())} of {P.height}")
    print(shared.head(8))

    print("\n== code pairs per map (2026 all plants): distinct pairs and top-pair coverage")
    pairs = (P.sort(["match_key", "round", "tick"]).group_by(["match_key", "map_name"])
             .agg(pl.col("site_code").unique().sort().alias("codes")))
    for mp in sorted(pairs["map_name"].unique().to_list()):
        g = pairs.filter(pl.col("map_name") == mp)
        keys = [str(tuple(c)) for c in g["codes"].to_list()]
        cnt = collections.Counter(keys)
        top = cnt.most_common(3)
        cover = sum(v for _, v in top) / len(keys)
        print(f"  {mp}: matches={len(keys)}; distinct code sets={len(cnt)}; top-3 sets cover {cover:.2f}; "
              f"top: {top[0][0] if top else ''} x{top[0][1] if top else 0}")

    print("\n== 2024 cross-era: site geometry and codes (fixed columns)")
    b24 = read24("bomb_state", ["match_key", "round", "tick", "event_type", "site_code",
                                "player_x_pos", "player_y_pos", "player_z_pos"])
    hk = "key" if "key" in h24_names else "match_key"
    h24 = read24("header", [hk, "map_name"]).rename({hk: "match_key"})
    pl24 = (b24.filter(pl.col("event_type") == "bomb_planted").join(h24, on="match_key", how="left"))
    print("2024 planted rows:", pl24.height, "matches:", pl24["match_key"].n_unique(),
          "null site_code:", int(pl24["site_code"].is_null().sum()))
    # 2026 centres from the descriptive clustering
    cent = {}
    P26 = P.filter(pl.col("kept") & pl.col("x").is_not_null()).unique(subset=["match_key", "round"], keep="first")
    P26 = P26.join(CL.select(["match_key", "round", "cluster"]), on=["match_key", "round"], how="inner")
    for (mp, cl), g in P26.group_by(["map_name", "cluster"]):
        cent[(mp, int(cl))] = g.select(["x", "y", "z"]).to_numpy().astype(np.float64).mean(0)
    for mp in sorted(P26["map_name"].unique().to_list()):
        sub = pl24.filter(pl.col("map_name") == mp)
        if sub.height == 0 or (mp, 0) not in cent or (mp, 1) not in cent:
            print(f"  {mp}: 2024 planted n={sub.height}; centres missing")
            continue
        X_all = sub.select(["player_x_pos", "player_y_pos", "player_z_pos"]).to_numpy().astype(np.float64)
        fin = np.isfinite(X_all).all(axis=1)
        print(f"  {mp}: 2024 planted rows with a null coordinate dropped: {int((~fin).sum())} of {len(X_all)}")
        X = X_all[fin]
        c0, c1 = cent[(mp, 0)], cent[(mp, 1)]
        d0 = np.linalg.norm(X - c0, axis=1)
        d1 = np.linalg.norm(X - c1, axis=1)
        dmin = np.minimum(d0, d1)
        print(f"  {mp}: 2024 planted n={len(X)}; share within 1000 of a 2026 centre={np.mean(dmin <= 1000):.3f}; "
              f"median distance to nearest centre={np.median(dmin):.0f}; share nearer c0={np.mean(d0 < d1):.3f}")
    ncodes = (pl24.filter(pl.col("event_type") == "bomb_planted")
              .group_by(["match_key"]).agg(pl.col("site_code").n_unique().alias("nc")))
    print(ncodes["nc"].value_counts().sort("nc"))

    print("\n== round-13 bomb-side mismatch by map and build (2024; 1 = exploded, 7 = defused)")
    rend = read24("round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "player_count"])
    h24b = read24("header", [hk, "map_name", "build_num"]).rename({hk: "match_key"})
    r13 = (rend.filter((pl.col("round") == 13) & (pl.col("player_count") == 20)
                       & pl.col("win_reason_code").is_in([1, 7]))
           .join(h24b, on="match_key", how="left"))
    r13 = r13.with_columns(
        (((pl.col("win_reason_code") == 1) & (pl.col("winner_team_code") != 2))
         | ((pl.col("win_reason_code") == 7) & (pl.col("winner_team_code") != 3))).alias("mismatch"))
    print("round 13 rows with a bomb outcome:", r13.height, "mismatches:", int(r13["mismatch"].sum()))
    print(r13.group_by(["build_num", "map_name"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
          .sort(["build_num", "map_name"]))


if __name__ == "__main__":
    main()
