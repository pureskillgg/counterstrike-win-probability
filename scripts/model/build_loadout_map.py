"""Per-slot loadout features for one map, aligned with the rows of data/round_state/<map>.npz.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\model\\build_loadout_map.py --map de_mirage

Slots and guards come from embed_round_state (the same rule that built the state arrays). For each day file, the position rows
of the map are joined to data/status_1hz on (match_key, round, second, player_id); the joined values are written to the slot
the position row occupies. Dead or missing slots stay zero. Missing status rows are counted, not filled with guesses.
Output: data/round_state/<map>_loadout.npy (N, 10, len(FEATURES)) float32 and <map>_loadout.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "model"))
import embed_round_state as ers  # noqa: E402

STATUS_DIR = ROOT / "data" / "status_1hz"
OUT = ROOT / "data" / "round_state"
FEATURES = ["health", "armor", "helmet", "defuser", "money", "equip_value", "defusing",
            "has_primary", "has_secondary", "has_flash", "has_smoke", "has_he_or_fire"]
STATUS_COLS = ["match_key", "round", "second", "player_id", "health", "armor", "has_helmet", "has_defuser", "money",
               "equipment_value_calc", "is_defusing", "inv_primary", "inv_secondary", "inv_flashbang",
               "inv_smokegrenade", "inv_hegrenade", "inv_molotov", "inv_incgrenade"]


def feature_exprs():
    f = lambda e: e.cast(pl.Float32)  # noqa: E731
    return [
        f(pl.col("health").fill_null(0) / 100).alias("health"),
        f(pl.col("armor").fill_null(0) / 100).alias("armor"),
        f(pl.col("has_helmet").fill_null(False)).alias("helmet"),
        f(pl.col("has_defuser").fill_null(False)).alias("defuser"),
        f(pl.col("money").fill_null(0) / 10000).alias("money"),
        f(pl.col("equipment_value_calc").fill_null(0) / 10000).alias("equip_value"),
        f(pl.col("is_defusing").fill_null(False)).alias("defusing"),
        f(pl.col("inv_primary").fill_null(0) > 0).alias("has_primary"),
        f(pl.col("inv_secondary").fill_null(0) > 0).alias("has_secondary"),
        f(pl.col("inv_flashbang").fill_null(0) > 0).alias("has_flash"),
        f(pl.col("inv_smokegrenade").fill_null(0) > 0).alias("has_smoke"),
        f((pl.col("inv_hegrenade").fill_null(0) > 0) | (pl.col("inv_molotov").fill_null(0) > 0)
          | (pl.col("inv_incgrenade").fill_null(0) > 0)).alias("has_he_or_fire"),
    ]


def main():
    import argparse
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="de_mirage")
    ap.add_argument("--in-stem", default=None, help="round-state npz stem in data/round_state/ (default: the map name)")
    ap.add_argument("--out-stem", default=None, help="output stem (default: the map name)")
    ap.add_argument("--maps-from", choices=["snapshot", "catalog"], default="snapshot",
                    help="match->map source: the 2026 snapshot cache (default) or data/combined/matches.parquet")
    a = ap.parse_args()
    map_name = a.map
    in_stem = a.in_stem or map_name
    out_stem = a.out_stem or map_name
    if a.maps_from == "catalog":
        import build_round_state_map as brs
        assign, match_maps = ers._defaults(None, brs.match_maps_from_catalog())
    else:
        assign, match_maps = ers._defaults(None, None)
    npz = np.load(OUT / f"{in_stem}.npz")
    keys = npz["match_keys"]
    m_idx = npz["match_idx"].astype(np.int64)
    rnd = npz["round"].astype(np.int64)
    sec = npz["second"].astype(np.int64)
    N = m_idx.size
    dkey = (m_idx * 128 + rnd) * 1024 + sec
    order = np.argsort(dkey, kind="stable")
    skey = dkey[order]
    match_lookup = pl.DataFrame({"match_key": keys, "match_idx": np.arange(keys.size, dtype=np.int64)})

    L = np.zeros((N, ers.N_SLOTS, len(FEATURES)), dtype=np.float32)
    matched = missing = 0
    day_paths = sorted(p for p in ers.POSITIONS_DIR.glob("*.parquet") if p.stem[:4].isdigit())
    print(f"map {map_name}: {len(day_paths)} day files, {N} state rows", flush=True)
    skipped = {}
    for p in day_paths:
        try:
            df, _ = ers._load_map_rows(map_name, [str(p)], assign, match_maps)
        except ValueError as exc:
            if "no labelled rows" not in str(exc):  # the side and duplicate-row guards are integrity failures, not skips
                raise
            skipped[p.name] = str(exc)  # a day with no labelled rows for the map (e.g. matches the match-map source does not hold)
            continue
        if df.height == 0:
            continue
        df = df.select(["match_key", "round", "second", "player_id", "slot"]).with_columns(
            pl.col("round").cast(pl.Int32), pl.col("second").cast(pl.Int32), pl.col("player_id").cast(pl.Int32))
        st = (pl.read_parquet(STATUS_DIR / p.name, columns=STATUS_COLS)
              .with_columns(pl.col("round").cast(pl.Int32), pl.col("second").cast(pl.Int32),
                            pl.col("player_id").cast(pl.Int32))
              .filter(pl.col("match_key").is_in(keys.tolist())))
        st = st.unique(["match_key", "round", "second", "player_id"], keep="first")
        j = df.join(st.select(["match_key", "round", "second", "player_id"] + [c for c in STATUS_COLS
                                                                            if c not in ("match_key", "round", "second", "player_id")]),
                    on=["match_key", "round", "second", "player_id"], how="left", coalesce=True)
        has_status = j["health"].is_not_null().to_numpy()
        missing += int((~has_status).sum())
        j = j.filter(pl.col("health").is_not_null())
        if j.height == 0:
            continue
        j = j.join(match_lookup, on="match_key", how="inner").with_columns(
            pl.col("match_idx").cast(pl.Int64), pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
        j = j.with_columns(((pl.col("match_idx") * 128 + pl.col("round")) * 1024 + pl.col("second")).alias("dkey"))
        feats = j.select(feature_exprs()).to_numpy().astype(np.float32)
        pos = np.searchsorted(skey, j["dkey"].to_numpy())
        ok = (pos < skey.size) & (skey[np.minimum(pos, skey.size - 1)] == j["dkey"].to_numpy())
        rows = order[np.minimum(pos, skey.size - 1)][ok]
        slots = j["slot"].to_numpy().astype(np.int64)[ok]
        L[rows, slots, :] = feats[ok]
        matched += int(ok.sum())
        print(f"  {p.name}: {int(ok.sum())} slot values written", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{out_stem}_loadout.npy", L)
    summary = {"map": map_name, "in_stem": in_stem, "out_stem": out_stem, "maps_from": a.maps_from, "features": FEATURES,
               "rows_expected": int(N), "shape": list(L.shape), "slot_values_written": matched,
               "position_rows_without_status": missing, "days_skipped": skipped, "elapsed_s": round(time.time() - t0, 1)}
    (OUT / f"{out_stem}_loadout.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
