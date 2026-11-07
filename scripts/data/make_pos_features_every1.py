"""Position, spacing and view-angle features per second, from data/positions_1hz + data/status_1hz + the grid.   (CPU, a few GB)

  ...python.exe -I -X utf8 scripts\\data\\make_pos_features.py [--store combined] [--every 2]   ->  data/<store>/cache_pos_2026.parquet

One row per (match_key, round, tick) of the grid (every --every-th second; 2 halves the size and the paired comparison uses the same seconds for both models).
All features are built from the living players only; an undefined value is null (XGBoost handles it). No map geometry or line of sight is used: distances are
straight lines in game units (about 250 u/s running speed), the view cone is a plain 90 degree wedge. Features (c = CT, t = T):
  site occupancy     {c,t}_site_hi / _lo   players within 1500 u of the busier / quieter bomb site (sorted, so site labels do not matter), {c,t}_nearest_site_mean
                     mean distance of a living player to the nearest site
  spacing            {c,t}_spread (mean distance to own centroid), ct_t_centroid_dist
  contact            min_enemy_dist, {c,t}_in_contact (players with an enemy within 1200 u)
  view               {c,t}_sees_enemy (players with an enemy inside their cone within 2000 u), {c,t}_exposed (players with an enemy within 1500 u that has them in its
                     cone while the player's own cone does not contain that enemy)
  blindness          {c,t}_blind_n (players still blinded > 0.25 s), {c,t}_blind_sum (seconds of blindness left, each capped at 4)
  bomb (planted)     c_min_bomb_dist, c_n_bomb_500 (CT within 500 u of the bomb), t_min_bomb_dist, defuse_slack (seconds the best-placed CT would have left to finish a
                     defuse: 40 - since plant - dist/250 - (5 with a kit, 10 without); needs a CT alive)
Site centres are k-means (k=2) of the bomb_begin_plant positions of each map (site codes are entity ids and change between builds, so they are not used).
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
SITE_R, CONTACT_R, SEE_R, EXPOSE_R, BOMB_R, RUN_SPEED = 1500.0, 1200.0, 2000.0, 1500.0, 500.0, 250.0
T, CT = 2, 3


def site_centres(store, match_keys=None):
    """Two plant-cluster centres per map. match_keys (optional): fit on these matches only, e.g. the train split."""
    P = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "_plants", "*.parquet")))], how="vertical_relaxed")
    M = pl.read_parquet(os.path.join(ROOT, "data", store, "matches.parquet")).select("match_key", "map_name")
    P = P.join(M, on="match_key").drop_nulls(["x", "y"])
    if match_keys is not None:
        P = P.filter(pl.col("match_key").is_in(list(match_keys)))
    out = {}
    for (mp,), g in P.group_by("map_name"):
        xy = np.stack([g["x"].to_numpy(), g["y"].to_numpy()], 1).astype("float64")
        rng = np.random.default_rng(0)
        best = None
        for _ in range(25):  # random restarts; keep the lowest inertia among splits where both sites hold at least 20% of the plants
            c = xy[rng.choice(len(xy), 2, replace=False)]
            for _ in range(40):
                d = ((xy[:, None, :] - c[None]) ** 2).sum(-1)
                lab = d.argmin(1)
                c = np.stack([np.median(xy[lab == k], 0) if (lab == k).any() else c[k] for k in range(2)])
            d = ((xy[:, None, :] - c[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            inertia = float(d.min(1).sum())
            share = min((lab == 0).mean(), (lab == 1).mean())
            if share >= 0.2 and (best is None or inertia < best[0]):
                best = (inertia, c.copy(), lab.copy())
        _, c, lab = best
        out[mp] = c[np.argsort(c[:, 0])]
        print(f"  {mp}: {len(xy)} plants, sites {np.round(out[mp]).tolist()}, sizes {[int((lab == k).sum()) for k in range(2)]}", flush=True)
    return out


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def scatter(df, kid, slot, col, n):
    a = np.full((n, 5), np.nan, "float32")
    a[df[kid].to_numpy(), df[slot].to_numpy()] = df[col].to_numpy()
    return a


def feats_for_day(pos, sta, G, planted_xy, sites, every, ev=None):
    i64 = lambda d: d.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    pos = i64(pos.filter(pl.col("second") % every == 0)).sort("tick")
    sta = i64(sta)
    if ev is not None:
        ev = i64(ev)
    if ev is not None and ev.height:
        e = ev.rename({"tick": "tick_e"}).select("match_key", "round", "player_id", "tick_e", "blind_duration").sort("tick_e")
        pos = pos.join_asof(e, left_on="tick", right_on="tick_e", by=["match_key", "round", "player_id"], strategy="backward")
        pos = pos.with_columns(((pl.col("tick_e") + pl.col("blind_duration") * 64 - pl.col("tick")) / 64).clip(0, None).fill_null(0.0).cast(pl.Float32).alias("blind_rem")).drop("tick_e", "blind_duration")
    else:
        pos = pos.with_columns(pl.lit(0.0).cast(pl.Float32).alias("blind_rem"))
    j = pos.join(sta.select("match_key", "round", "tick", "player_id", "health", "has_defuser", "is_defusing"), on=["match_key", "round", "tick", "player_id"], how="left")
    K = j.select("match_key", "round", "tick").unique(maintain_order=True).with_row_index("kid")
    n = K.height
    j = j.join(K, on=["match_key", "round", "tick"]).sort("kid", "side", "player_id")
    j = j.with_columns(pl.int_range(pl.len()).over(["kid", "side"]).alias("slot")).filter(pl.col("slot") < 5)
    A = {}
    for nm, sd in (("c", CT), ("t", T)):
        s = j.filter(pl.col("side") == sd).with_columns(pl.col("has_defuser").cast(pl.Float32).fill_null(0.0), pl.col("is_defusing").cast(pl.Float32).fill_null(0.0))
        A[nm] = {c: scatter(s, "kid", "slot", c, n) for c in ("x", "y", "yaw", "has_defuser", "blind_rem")}
    KG = K.join(G, on=["match_key", "round", "tick"], how="left")
    # bomb position per tick: the latest bomb_begin_plant of the same round at or before that tick (null before the first one).
    # Backward asof on the tick, so a later plant can never reach an earlier second.
    KB = K.sort("tick").join_asof(planted_xy.sort("tick"), on="tick", by=["match_key", "round"], strategy="backward").sort("kid")
    sp = KG["seconds_since_plant"].fill_null(0).to_numpy().astype("float32")
    planted = (KG["bomb_planted"].fill_null(0).to_numpy() == 1)
    mp = KG["map_name"].to_list()
    SA = np.array([sites.get(m, np.full((2, 2), np.nan))[0] for m in mp], "float32")
    SB = np.array([sites.get(m, np.full((2, 2), np.nan))[1] for m in mp], "float32")
    bx, by = KB["bx"].to_numpy().astype("float32"), KB["by"].to_numpy().astype("float32")
    out = {"match_key": K["match_key"], "round": K["round"], "tick": K["tick"]}
    nanmean = lambda a: np.where(np.isfinite(a).any(1), np.nanmean(np.where(np.isfinite(a), a, np.nan), 1), np.nan)

    def dist(x, y, cx, cy):
        return np.sqrt((x - cx[:, None]) ** 2 + (y - cy[:, None]) ** 2)

    with np.errstate(all="ignore"):
        for nm in ("c", "t"):
            x, y = A[nm]["x"], A[nm]["y"]
            dA, dB = dist(x, y, SA[:, 0], SA[:, 1]), dist(x, y, SB[:, 0], SB[:, 1])
            na, nb = np.nansum(dA < SITE_R, 1), np.nansum(dB < SITE_R, 1)
            out[f"{nm}_site_hi"], out[f"{nm}_site_lo"] = np.maximum(na, nb).astype("float32"), np.minimum(na, nb).astype("float32")
            out[f"{nm}_nearest_site_mean"] = nanmean(np.minimum(dA, dB))
            cx, cy = nanmean(x), nanmean(y)
            out[f"{nm}_spread"] = nanmean(dist(x, y, cx, cy))
            A[nm]["cx"], A[nm]["cy"] = cx, cy
            alive_n = np.isfinite(x).sum(1)
            out[f"{nm}_blind_n"] = np.where(alive_n > 0, np.nansum(A[nm]["blind_rem"] > 0.25, 1), np.nan).astype("float32")
            out[f"{nm}_blind_sum"] = np.where(alive_n > 0, np.nansum(np.minimum(A[nm]["blind_rem"], 4.0), 1), np.nan).astype("float32")
        out["ct_t_centroid_dist"] = np.sqrt((A["c"]["cx"] - A["t"]["cx"]) ** 2 + (A["c"]["cy"] - A["t"]["cy"]) ** 2)
        # pairwise CT i - T j: distance, bearing CT->T
        dx = A["t"]["x"][:, None, :] - A["c"]["x"][:, :, None]
        dy = A["t"]["y"][:, None, :] - A["c"]["y"][:, :, None]
        D = np.sqrt(dx ** 2 + dy ** 2)  # [n, ct i, t j]
        brg = np.degrees(np.arctan2(dy, dx))  # bearing from CT i to T j
        c_in_t = np.abs(wrap(A["c"]["yaw"][:, :, None] - brg)) < 45.0  # T j inside CT i's cone
        t_in_c = np.abs(wrap(A["t"]["yaw"][:, None, :] - (brg + 180.0))) < 45.0  # CT i inside T j's cone
        out["min_enemy_dist"] = np.nanmin(np.where(np.isfinite(D), D, np.inf), axis=(1, 2)).astype("float32")
        out["min_enemy_dist"] = np.where(np.isfinite(out["min_enemy_dist"]), out["min_enemy_dist"], np.nan)
        out["c_in_contact"] = np.nansum((D < CONTACT_R).any(2) & np.isfinite(D).any(2), 1).astype("float32")
        out["t_in_contact"] = np.nansum((D < CONTACT_R).any(1) & np.isfinite(D).any(1), 1).astype("float32")
        out["c_sees_enemy"] = (((D < SEE_R) & c_in_t).any(2)).sum(1).astype("float32")
        out["t_sees_enemy"] = (((D < SEE_R) & t_in_c).any(1)).sum(1).astype("float32")
        out["c_exposed"] = ((((D < EXPOSE_R) & t_in_c & ~c_in_t)).any(2)).sum(1).astype("float32")
        out["t_exposed"] = ((((D < EXPOSE_R) & c_in_t & ~t_in_c)).any(1)).sum(1).astype("float32")
        hasb = planted & np.isfinite(bx)
        for nm, pre in (("c", "c"), ("t", "t")):
            db = dist(A[nm]["x"], A[nm]["y"], bx, by)
            db = np.where(hasb[:, None], db, np.nan)
            out[f"{pre}_min_bomb_dist"] = np.where(np.isfinite(db).any(1), np.nanmin(np.where(np.isfinite(db), db, np.inf), 1), np.nan).astype("float32")
            if nm == "c":
                out["c_n_bomb_500"] = np.where(hasb, np.nansum(db < BOMB_R, 1), np.nan).astype("float32")
                need = db / RUN_SPEED + np.where(A["c"]["has_defuser"] > 0, 5.0, 10.0)
                slack = 40.0 - sp[:, None] - need
                out["defuse_slack"] = np.where(np.isfinite(slack).any(1), np.nanmax(np.where(np.isfinite(slack), slack, -np.inf), 1), np.nan).astype("float32")
    df = pl.DataFrame({k: (v if isinstance(v, pl.Series) else pl.Series(k, np.asarray(v, "float32"))) for k, v in out.items()})
    return df.with_columns([pl.col(c).fill_nan(None) for c in df.columns if df.schema[c] == pl.Float32])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--centre_keys", default=None, help="json list of match keys: fit the site centres on these matches only (writes *_trainctr outputs in their own folders)")
    a = ap.parse_args()
    d = os.path.join(ROOT, "data", a.store)
    tag = ""
    centre_keys = None
    if a.centre_keys:
        import json
        centre_keys = json.load(open(a.centre_keys))
        tag = "_trainctr"
        print(f"site centres from {len(centre_keys)} matches in {a.centre_keys}", flush=True)
    print("site centres:", flush=True)
    sites = site_centres(a.store, centre_keys)
    G = pl.read_parquet(os.path.join(d, "grid.parquet"), columns=["match_key", "round", "tick", "bomb_planted", "seconds_since_plant", "map_name", "era", "second"]).filter(pl.col("era") == "2026")
    PL = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "_plants", "*.parquet")))], how="vertical_relaxed")
    # one row per bomb_begin_plant event (tick, position), not one per round: the per-tick lookup in feats_for_day needs the tick
    planted_xy = (PL.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).sort("tick", maintain_order=True)
                  .unique(["match_key", "round", "tick"], keep="last", maintain_order=True)
                  .select("match_key", "round", "tick", pl.col("x").alias("bx"), pl.col("y").alias("by")))
    parts = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "2026-*.parquet"))):
        day = os.path.basename(fp)
        fs = os.path.join(ROOT, "data", "status_1hz", day)
        if not os.path.exists(fs):
            print("skip (no status)", day)
            continue
        fb = os.path.join(ROOT, "data", "blind_events", day)
        fd = os.path.join(ROOT, "data", "pos_features_1hz", f"every{a.every}{tag}", day)
        os.makedirs(os.path.dirname(fd), exist_ok=True)
        newest_in = max(os.path.getmtime(x) for x in (fp, fs, fb, os.path.join(ROOT, "scripts", "data", "make_pos_features.py")) if os.path.exists(x))
        if os.path.exists(fd) and os.path.getmtime(fd) >= newest_in:  # nothing this day depends on changed: reuse its feature file (a new match changes the day's input files)
            parts.append(pl.read_parquet(fd))
            print(f"{day}: reused ({parts[-1].height:,} rows)", flush=True)
            continue
        pos = pl.read_parquet(fp).with_columns(pl.col("round").cast(pl.Int64))
        sta = pl.read_parquet(fs).with_columns(pl.col("round").cast(pl.Int64))
        mk = pos["match_key"].unique().to_list()
        Gd = G.filter(pl.col("match_key").is_in(mk)).select("match_key", "round", "tick", "bomb_planted", "seconds_since_plant", "map_name")
        ev = pl.read_parquet(fb) if os.path.exists(fb) else None
        if ev is None:
            print("   no blind events for", day, flush=True)
        # the day is computed in blocks of matches: the features are per match, so the result is the same, and the peak memory is a fraction of a day
        f_parts = []
        for ci in range(0, len(mk), 300):
            cm = mk[ci:ci + 300]
            f_parts.append(feats_for_day(pos.filter(pl.col("match_key").is_in(cm)), sta.filter(pl.col("match_key").is_in(cm)),
                                         Gd.filter(pl.col("match_key").is_in(cm)), planted_xy.filter(pl.col("match_key").is_in(cm)), sites, a.every,
                                         ev.filter(pl.col("match_key").is_in(cm)) if ev is not None else None))
        f = pl.concat(f_parts, how="vertical_relaxed")
        f.write_parquet(fd + ".tmp", compression="zstd")
        os.replace(fd + ".tmp", fd)
        parts.append(f)
        print(f"{day}: {f.height:,} rows, {len(mk)} matches", flush=True)
    out = pl.concat(parts, how="vertical_relaxed").with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    fn = os.path.join(d, f"cache_pos_2026{tag}_every{a.every}.parquet")  # patched: every-second build keeps the every-2 file intact
    out.write_parquet(fn + ".tmp", compression="zstd")
    os.replace(fn + ".tmp", fn)
    print(f"-> {fn}: {out.height:,} rows x {out.width} columns")
    desc = out.drop("match_key", "round", "tick").describe()
    print(desc)


if __name__ == "__main__":
    main()
