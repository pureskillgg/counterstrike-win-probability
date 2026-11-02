"""Per-player tensors for a graph / set model of one second of a round.   (CPU, mmap output; a few GB)

  ...python.exe -I -X utf8 scripts\\model\\gnn\\make_player_tensors.py [--store combined]

Rows = exactly the rows of the position experiment (exp_positions.py): the inner join of the snapshot cache and the position feature cache (every other second of every
used 2026 round), ordered by upload day, match, round, tick. Writes under data/gnn/:
  index.parquet     rid, match_key, round, tick, second, label_ct_win, w_round, map_name, upload_day, prev1, prev2 (row id of the sampled second 2 / 4 s earlier in the same round, -1 if none)
  G59.npy           [N, 59] float32  the final model's inputs (alive counts, clock, bomb, map one-hots, loadout totals), column order = models/winprob_v3.meta.json features
  GLOB.npy          [N, 19] float32  clock / bomb / map / site geometry in radar units (see GLOB_NAMES)
  P.npy             [N, 10, F] float16  players: slots 0-4 = CT, 5-9 = T, ordered by player id; dead players are all zero with MASK = 0
  MASK.npy          [N, 10] uint8
  meta.json         feature names, shapes, counts
Player features (F = 29, see P_NAMES): radar position, z, view direction, hp, armor, helmet, kit, defusing, c4, weapon class one-hot, grenades held, money, equipment value,
blindness left, distance to the planted bomb and to the nearest bomb site. All from the 1 Hz tables (positions_1hz, status_1hz, blind_events): nothing from the future of the row's tick.
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "data"))
sys.path.insert(0, os.path.join(HERE, ".."))
import make_pos_features as MPF  # noqa: E402  (site centres)
import wp_items as WI  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]
P_NAMES = ["px", "py", "z", "yaw_sin", "yaw_cos", "hp", "armor", "helmet", "kit", "defusing", "c4", "w_rifle", "w_awp", "w_sniper_other", "w_smg", "w_heavy", "w_none",
           "flash", "smoke", "he", "fire", "decoy", "money", "equip", "blind", "dist_bomb", "dist_site", "has_secondary", "alive"]
GLOB_NAMES = ["sec_live", "planted", "since_plant", "alive_ct", "alive_t"] + [f"map_{m}" for m in MAPS] + ["bomb_px", "bomb_py", "siteA_px", "siteA_py", "siteB_px", "siteB_py", "wscale"]
F, NG = len(P_NAMES), len(GLOB_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "gnn"))
    ap.add_argument("--from-day", default="", help="resume: keep the P / MASK memmaps and rebuild only the upload days >= this one (yyyy/mm/dd); the row order is deterministic")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = os.path.join(ROOT, "data", a.store)
    t0 = time.time()
    meta_f = json.load(open(os.path.join(ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    maps_meta = json.load(open(os.path.join(ROOT, "notes", "replay", "map_meta.json"), encoding="utf-8"))
    sites_world = MPF.site_centres(a.store)

    # ---- row index = snapshot cache x position cache (same rows as exp_positions.py)
    keys = ["match_key", "round", "tick"]
    pos = pl.scan_parquet(os.path.join(d, "cache_pos_2026.parquet")).select(keys)
    cols = list(dict.fromkeys(keys + ["second", "label_ct_win", "w_round", "map_name"] + meta_f))
    snap = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(cols)
    M = pl.read_parquet(os.path.join(d, "matches.parquet")).select("match_key", "upload_day")
    idx = snap.join(pos, on=keys, how="inner").collect().join(M, on="match_key", how="left")
    idx = idx.sort(["upload_day", "match_key", "round", "tick"]).with_row_index("rid")
    N = idx.height
    print(f"rows {N:,}; matches {idx['match_key'].n_unique():,}; days {idx['upload_day'].n_unique()} ({time.time() - t0:.0f}s)", flush=True)
    G59 = idx.select(meta_f).to_numpy().astype(np.float32)
    np.save(os.path.join(a.out, "G59.npy"), G59)
    gl_cols = {n: G59[:, meta_f.index(n)].copy() for n in ("seconds_since_round_live", "bomb_planted", "seconds_since_plant", "alive_ct", "alive_t")}
    del G59
    idx = idx.select("rid", "match_key", "round", "tick", "second", "label_ct_win", "w_round", "map_name", "upload_day")
    # previous sampled seconds of the same round (rows are every other second, so prev1 is 2 s earlier)
    mk = idx["match_key"].to_numpy()
    rd = idx["round"].to_numpy().astype(np.int64)
    sec = idx["second"].to_numpy().astype(np.int64)
    same1 = np.zeros(N, bool)
    same1[1:] = (mk[1:] == mk[:-1]) & (rd[1:] == rd[:-1]) & (sec[1:] - sec[:-1] == 2)
    prev1 = np.where(same1, np.arange(N) - 1, -1)
    prev2 = np.where(prev1 >= 0, prev1[np.maximum(prev1, 0)], -1)  # the row 4 s earlier = the previous row of the previous row
    idx = idx.with_columns(pl.Series("prev1", prev1.astype(np.int64)), pl.Series("prev2", prev2.astype(np.int64)))
    idx.select("rid", "match_key", "round", "tick", "second", "label_ct_win", "w_round", "map_name", "upload_day", "prev1", "prev2").write_parquet(os.path.join(a.out, "index.parquet"))

    # ---- map geometry in radar units
    def radar(mp, x, y):
        m = maps_meta[mp]
        return (x - m["pos_x"]) / m["scale"] / 1024.0, (m["pos_y"] - y) / m["scale"] / 1024.0

    site_r = {}
    for mp, c in sites_world.items():
        (ax, ay), (bx, by) = radar(mp, c[0, 0], c[0, 1]), radar(mp, c[1, 0], c[1, 1])
        site_r[mp] = (ax, ay, bx, by)
    map_names = idx["map_name"].to_numpy()
    GL = np.zeros((N, NG), np.float32)
    GL[:, 0] = gl_cols["seconds_since_round_live"] / 115.0
    GL[:, 1] = gl_cols["bomb_planted"]
    GL[:, 2] = np.nan_to_num(gl_cols["seconds_since_plant"]) / 40.0
    GL[:, 3] = gl_cols["alive_ct"] / 5.0
    GL[:, 4] = gl_cols["alive_t"] / 5.0
    wscale = np.zeros(N, np.float32)
    for j, m in enumerate(MAPS):
        sel = map_names == f"de_{m}"
        GL[sel, 5 + j] = 1.0
        s = site_r.get(f"de_{m}", (0, 0, 0, 0))
        GL[sel, 14], GL[sel, 15], GL[sel, 16], GL[sel, 17] = s
        wscale[sel] = maps_meta[f"de_{m}"]["scale"] * 1024.0 / 3000.0
    GL[:, 18] = wscale
    # planted bomb position (last begin_plant of the round), radar units
    PL = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "_plants", "*.parquet")))], how="vertical_relaxed")
    pxy = PL.sort("tick").group_by("match_key", "round", maintain_order=True).agg(pl.col("x").last().alias("bx"), pl.col("y").last().alias("by")).with_columns(pl.col("round").cast(pl.Int64))
    bxy = idx.select("match_key", "round").join(pxy, on=["match_key", "round"], how="left")
    bx, by = bxy["bx"].to_numpy().astype(np.float32), bxy["by"].to_numpy().astype(np.float32)
    for j, m in enumerate(MAPS):
        sel = (map_names == f"de_{m}") & (GL[:, 1] > 0) & np.isfinite(bx)
        if sel.any():
            GL[sel, 12], GL[sel, 13] = radar(f"de_{m}", bx[sel], by[sel])
    GL[:, 12] = np.where(np.isfinite(GL[:, 12]) & (GL[:, 1] > 0), GL[:, 12], 0.0)
    GL[:, 13] = np.where(np.isfinite(GL[:, 13]) & (GL[:, 1] > 0), GL[:, 13], 0.0)
    GL[:, 14:] = np.nan_to_num(GL[:, 14:])
    np.save(os.path.join(a.out, "GLOB.npy"), GL)

    # ---- players, day by day
    mode = "r+" if a.from_day else "w+"
    Pm = np.lib.format.open_memmap(os.path.join(a.out, "P.npy"), mode=mode, dtype=np.float16, shape=(N, 10, F))
    Mm = np.lib.format.open_memmap(os.path.join(a.out, "MASK.npy"), mode=mode, dtype=np.uint8, shape=(N, 10))
    days = sorted(idx["upload_day"].unique().to_list())
    idx_small = idx.select("rid", "match_key", "round", "tick", "map_name", "upload_day")
    bomb_world = {}
    n_players = 0
    for day in days:
        if a.from_day and day < a.from_day:
            continue
        fname = day.replace("/", "-") + ".parquet"
        fp, fs, fb = (os.path.join(ROOT, "data", "positions_1hz", fname), os.path.join(ROOT, "data", "status_1hz", fname), os.path.join(ROOT, "data", "blind_events", fname))
        ix = idx_small.filter(pl.col("upload_day") == day).select("rid", "match_key", "round", "tick", "map_name")
        if ix.height == 0 or not os.path.exists(fp) or not os.path.exists(fs):
            continue
        i64 = lambda df: df.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
        ps = i64(pl.read_parquet(fp)).drop("second")
        st = i64(pl.read_parquet(fs)).drop("second", "side")
        j = ix.join(ps, on=["match_key", "round", "tick"], how="inner").join(st, on=["match_key", "round", "tick", "player_id"], how="left")
        if os.path.exists(fb):
            ev = i64(pl.read_parquet(fb)).rename({"tick": "tick_e"}).select("match_key", "round", "player_id", "tick_e", "blind_duration").sort("tick_e")
            j = j.sort("tick").join_asof(ev, left_on="tick", right_on="tick_e", by=["match_key", "round", "player_id"], strategy="backward")
            j = j.with_columns(((pl.col("tick_e") + pl.col("blind_duration") * 64 - pl.col("tick")) / 64).clip(0, 4).fill_null(0.0).alias("blind_rem")).drop("tick_e", "blind_duration")
        else:
            j = j.with_columns(pl.lit(0.0).alias("blind_rem"))
        j = j.sort("rid", "side", "player_id").with_columns(pl.int_range(pl.len()).over(["rid", "side"]).alias("rank")).filter(pl.col("rank") < 5)
        j = j.with_columns(pl.when(pl.col("side") == 3).then(pl.col("rank")).otherwise(pl.col("rank") + 5).alias("slot"))
        rid = j["rid"].to_numpy()
        slot = j["slot"].to_numpy()
        n = j.height
        X = np.zeros((n, F), np.float32)
        mp = j["map_name"].to_numpy()
        xx, yy, zz, yaw = (j[c].to_numpy().astype(np.float32) for c in ("x", "y", "z", "yaw"))
        px = np.zeros(n, np.float32)
        py = np.zeros(n, np.float32)
        wsc = np.ones(n, np.float32)
        for m in MAPS:
            sel = mp == f"de_{m}"
            if sel.any():
                px[sel], py[sel] = radar(f"de_{m}", xx[sel], yy[sel])
                wsc[sel] = maps_meta[f"de_{m}"]["scale"] * 1024.0
        X[:, 0], X[:, 1], X[:, 2] = px, py, zz / 400.0
        X[:, 3], X[:, 4] = np.sin(np.radians(yaw)), np.cos(np.radians(yaw))
        X[:, 5] = j["health"].fill_null(0).to_numpy() / 100.0
        X[:, 6] = j["armor"].fill_null(0).to_numpy() / 100.0
        X[:, 7] = j["has_helmet"].fill_null(False).to_numpy().astype(np.float32)
        X[:, 8] = j["has_defuser"].fill_null(False).to_numpy().astype(np.float32)
        X[:, 9] = j["is_defusing"].fill_null(False).to_numpy().astype(np.float32)
        X[:, 10] = (j["inv_c4"].fill_null(0).to_numpy() > 0).astype(np.float32)
        ip = j["inv_primary"].fill_null(0).to_numpy()
        for c, grp in ((11, WI.RIFLE), (12, WI.AWP), (13, WI.SNIPER_OTHER), (14, WI.SMG), (15, WI.HEAVY)):
            X[:, c] = np.isin(ip, grp).astype(np.float32)
        X[:, 16] = (ip == 0).astype(np.float32)
        for c, nm in ((17, "inv_flashbang"), (18, "inv_smokegrenade"), (19, "inv_hegrenade"), (21, "inv_decoy")):
            X[:, c] = np.minimum(j[nm].fill_null(0).to_numpy(), 2).astype(np.float32)
        X[:, 20] = np.minimum(j["inv_molotov"].fill_null(0).to_numpy() + j["inv_incgrenade"].fill_null(0).to_numpy(), 2).astype(np.float32)
        X[:, 22] = j["money"].fill_null(0).to_numpy() / 10000.0
        X[:, 23] = j["equipment_value_calc"].fill_null(0).to_numpy() / 10000.0
        X[:, 24] = j["blind_rem"].to_numpy().astype(np.float32) / 4.0
        X[:, 27] = (j["inv_secondary"].fill_null(0).to_numpy() > 0).astype(np.float32)
        gl = GL[rid]
        planted = gl[:, 1] > 0
        d_bomb = np.sqrt((px - gl[:, 12]) ** 2 + (py - gl[:, 13]) ** 2) * wsc / 2000.0
        X[:, 25] = np.where(planted, np.minimum(d_bomb, 3.0), 0.0)
        dA = np.sqrt((px - gl[:, 14]) ** 2 + (py - gl[:, 15]) ** 2)
        dB = np.sqrt((px - gl[:, 16]) ** 2 + (py - gl[:, 17]) ** 2)
        X[:, 26] = np.minimum(np.minimum(dA, dB) * wsc / 2000.0, 3.0)
        X[:, 28] = 1.0
        X = np.nan_to_num(X)
        # scatter into the memmaps (rows of one day are contiguous in rid order)
        r0, r1 = int(ix["rid"].min()), int(ix["rid"].max()) + 1
        blk = np.zeros((r1 - r0, 10, F), np.float16)
        mblk = np.zeros((r1 - r0, 10), np.uint8)
        blk[rid - r0, slot] = X.astype(np.float16)
        mblk[rid - r0, slot] = 1
        Pm[r0:r1] = blk
        Mm[r0:r1] = mblk
        n_players += n
        print(f"{day}: {ix.height:,} rows, {n:,} player rows ({time.time() - t0:.0f}s)", flush=True)
    Pm.flush()
    Mm.flush()
    json.dump({"rows": int(N), "P_shape": [int(N), 10, F], "P_names": P_NAMES, "GLOB_names": GLOB_NAMES, "G59_names": meta_f, "player_rows": int(n_players),
               "built": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1)
    print(f"done: {N:,} rows, {n_players:,} player rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
