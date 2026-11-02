"""Build the per-second round-state arrays for one map, one day file at a time.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\model\\build_round_state_map.py --map de_mirage

Two passes over the day files. Pass 1 fits the scaling window from TRAIN rows one day at a time; the minimum and maximum
combine across days. Pass 2 builds each day with that one window and stacks the results. Only one day is loaded at a time,
because stacking all days of one map at once needs more memory than this machine has free next to other work.

Writes data/round_state/<map>.npz with X (N, 10, 6 float32, slots 0-4 CT and 5-9 T), split (0 train, 1 val, 2 test),
match_idx (index into match_keys), round and second (int16), and match_keys; and data/round_state/<map>.json with the
window, the counts and the per-split out-of-window fractions. Nothing is written under tomes/, models/ or data/combined/.
"""
import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_round_state as ers  # noqa: E402

OUT_DIR = ers.ROOT / "data" / "round_state"
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
SUM_KEYS = (
    "position_rows_in_map",
    "position_rows_unmapped_match_key",
    "rows_dropped_sentinel_pid",
    "rows_player_id_0_kept",
    "rows_dropped_bad_coordinates",
    "rows_dropped_roster_overflow",
    "rows_dropped_unlabelled",
    "n_rows",
    "rounds_gapped",
)
DAY_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.parquet$")


def day_files(dirs=None):
    dirs = [Path(d) for d in (dirs or [ers.POSITIONS_DIR])]
    names = {}
    for d in dirs:
        for p in d.glob("*.parquet"):
            if DAY_NAME.match(p.name):
                if p.name in names:
                    raise ValueError(f"two positions folders hold {p.name}")
                names[p.name] = p
    return [names[n] for n in sorted(names)]


def match_maps_from_catalog():
    """match_key -> map_name from data/combined/matches.parquet (covers every era, unlike the 2026 snapshot cache)."""
    import polars as pl
    m = pl.read_parquet(ers.ROOT / "data" / "combined" / "matches.parquet", columns=["match_key", "map_name"]).unique()
    if m["match_key"].n_unique() != m.height:
        raise ValueError("matches.parquet has a match_key with more than one map_name")
    return dict(zip(m["match_key"].to_list(), m["map_name"].to_list()))


def fixed_window(map_name):
    """The window already in data/round_state/<map>.json (fitted on the original train rows), kept fixed for every new row."""
    meta = json.loads((OUT_DIR / f"{map_name}.json").read_text(encoding="utf-8"))
    w = dict(meta["window"])
    return {k: float(w[k]) for k in ("xy_min", "xy_max", "z_min", "z_max")}, {}


def fit_window(map_name, days, assign, match_maps):
    """Pass 1: the train-only window, combined over days. Days without train rows for the map are skipped and listed."""
    lo = {"xy_min": np.inf, "xy_max": -np.inf, "z_min": np.inf, "z_max": -np.inf}
    n_train = 0
    per_day = {}
    for day in days:
        try:
            b = ers.fit_map_bounds(map_name, [day], assign=assign, match_maps=match_maps)
        except ValueError as exc:
            per_day[day.name] = f"skipped: {exc}"
            continue
        lo["xy_min"] = min(lo["xy_min"], b["xy_min"])
        lo["xy_max"] = max(lo["xy_max"], b["xy_max"])
        lo["z_min"] = min(lo["z_min"], b["z_min"])
        lo["z_max"] = max(lo["z_max"], b["z_max"])
        n_train += b["n_train_rows"]
        per_day[day.name] = b["n_train_rows"]
    lo = {k: float(v) for k, v in lo.items()}
    lo["n_train_rows"] = n_train
    return lo, per_day


def build_days(map_name, days, bounds, assign, match_maps):
    """Pass 2: one day at a time with the fixed window, then stacked."""
    parts = {"X": [], "split": [], "match_key": [], "round": [], "second": []}
    stats_list = []
    skipped = {}
    for day in days:
        try:
            ds = ers.build_map_dataset(map_name, [day], bounds=bounds, assign=assign, match_maps=match_maps)
        except ValueError as exc:
            skipped[day.name] = str(exc)
            continue
        parts["X"].append(ds.X)
        parts["split"].append(np.array([SPLIT_CODE[s] for s in ds.split], dtype=np.int8))
        parts["match_key"].append(ds.match_key)
        parts["round"].append(ds.round.astype(np.int16))
        parts["second"].append(ds.second.astype(np.int16))
        stats_list.append(ds.stats)
        print(f"  {day.name}: {ds.X.shape[0]} rows, {len(set(ds.match_key.tolist()))} matches", flush=True)
    X = np.concatenate(parts["X"])
    split = np.concatenate(parts["split"])
    match_key = np.concatenate(parts["match_key"])
    match_keys, match_idx = np.unique(match_key, return_inverse=True)
    return {
        "X": X,
        "split": split,
        "match_keys": match_keys,
        "match_idx": match_idx.astype(np.int32),
        "round": np.concatenate(parts["round"]),
        "second": np.concatenate(parts["second"]),
        "stats_list": stats_list,
        "skipped": skipped,
    }


def out_of_window(X, split):
    """Fraction of alive-slot coordinate values (x, y, z) outside [0, 1], per split. Recomputed from X, not averaged over days."""
    alive = X[:, :, 0] == 1
    result = {}
    for name, code in SPLIT_CODE.items():
        m = split == code
        if not m.any():
            continue
        a = alive[m]
        c = X[m][:, :, 1:4]
        bad = ((c < 0) | (c > 1)) & a[:, :, None]
        result[name] = float(bad.sum() / (a.sum() * 3))
    return result


def main():
    warnings.filterwarnings("ignore", message="scaling window fitted from these day_files only")
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--max-days", type=int, default=0, help="for a quick run: only the first N day files (0 = all)")
    ap.add_argument("--day-dir", action="append", default=None, help="positions folder (repeatable); default data/positions_1hz")
    ap.add_argument("--fixed-window", action="store_true", help="keep the window already in data/round_state/<map>.json (no refit)")
    ap.add_argument("--maps-from", choices=["snapshot", "catalog"], default="snapshot",
                    help="match->map source: the 2026 snapshot cache (default) or data/combined/matches.parquet (all eras)")
    ap.add_argument("--out-name", default=None, help="output file stem in data/round_state/ (default: the map name)")
    a = ap.parse_args()
    out_stem = a.out_name or a.map

    t0 = time.perf_counter()
    match_maps = match_maps_from_catalog() if a.maps_from == "catalog" else None
    assign, match_maps = ers._defaults(None, match_maps)
    days = day_files(a.day_dir)
    if a.max_days:
        days = days[: a.max_days]
    print(f"map {a.map}: {len(days)} day files, {len(assign)} manifest entries, maps from {a.maps_from}", flush=True)

    if a.fixed_window:
        bounds, per_day_train = fixed_window(a.map)
        print(f"window (fixed, from {a.map}.json): {bounds}", flush=True)
    else:
        bounds, per_day_train = fit_window(a.map, days, assign, match_maps)
        print(f"window (train rows, all days): {bounds}", flush=True)
    print(f"pass 1 done in {time.perf_counter() - t0:.0f} s", flush=True)

    built = build_days(a.map, days, bounds, assign, match_maps)
    X, split = built["X"], built["split"]
    stats = {k: int(sum(s.get(k, 0) for s in built["stats_list"])) for k in SUM_KEYS}
    stats["frac_alive_slots"] = float(X[:, :, 0].mean())
    stats["rows_per_split"] = {s: int((split == c).sum()) for s, c in SPLIT_CODE.items()}
    stats["matches_per_split"] = {
        s: int(np.unique(built["match_idx"][split == c]).size) for s, c in SPLIT_CODE.items()
    }
    stats["frac_alive_coords_outside_window"] = out_of_window(X, split)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz = OUT_DIR / f"{out_stem}.npz"
    np.savez(
        npz,
        X=X,
        split=split,
        match_idx=built["match_idx"],
        round=built["round"],
        second=built["second"],
        match_keys=built["match_keys"],
    )
    summary = {
        "map": a.map,
        "split_codes": SPLIT_CODE,
        "window": bounds,
        "window_train_rows_by_day": per_day_train,
        "days_built": len(built["stats_list"]),
        "days_skipped": built["skipped"],
        "stats": stats,
        "X_shape": list(X.shape),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    summary["day_dirs"] = a.day_dir or [str(ers.POSITIONS_DIR)]
    summary["maps_from"] = a.maps_from
    summary["fixed_window"] = bool(a.fixed_window)
    (OUT_DIR / f"{out_stem}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("window", "days_built", "X_shape", "elapsed_s")}, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"wrote {npz} ({npz.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
