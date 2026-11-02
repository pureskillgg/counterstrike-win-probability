"""Per-second round-state arrays for one map, the input to a round-state embedding.

What this file does
  For one map, builds one row per (match_key, round, second) that has at least one
  living player in the 1 Hz position files. Each row is a fixed (10, 6) float32 array:
    - slots 0-4 are CT players, slots 5-9 are T players. Within a side, the slot is the
      rank of player_id among that side's players in the same round (ascending).
    - channels are alive, x, y, z, sin(yaw), cos(yaw).
    - missing or dead slots are all zeros, with alive = 0.
  Coordinates are min-max scaled with a window fitted on TRAIN rows only. x and y share
  one window (the horizontal extent); z has its own window. The window is returned, so
  every day and split can reuse the same one.
  Each row carries its split label from the manifest.
  Also provides a numpy PCA baseline (fitted on train rows only) and the signature of a
  DeepSets encoder with latent size 8. The encoder is not implemented, because torch is
  not installed in .pixi/envs/default.

Inputs (read only)
  data/positions_1hz/<yyyy-mm-dd>.parquet
      match_key, round, tick, second, player_id, side (2 = T, 3 = CT), x, y, z, yaw (degrees)
  data/combined/cache_snapshot_2026.parquet   match_key and map_name only
  data/combined/split_manifest.json           {"assign": {match_key: "train" | "val" | "test"}}

Guards
  - no match_key may belong to two splits (checked on the manifest and on the output rows)
  - each match_key must have exactly one map_name in the snapshot
  - duplicate (match_key, round, second, player_id) rows raise
  - player_id 65535 (sentinel) rows are dropped and counted
  - rows at the exact origin (x = y = z = 0), with any coordinate of magnitude 1e4 or more (the -16384 z sentinel),
    or non-finite, are recording gaps: dropped and counted before the scaling window is fitted (the sentinel rows
    set z_min to -16384 before this rule; see the verification record for the 2026-10-22 check)
  - position rows whose match_key is missing from cache_snapshot_2026 are counted (they belong to no map)
  - player_id 0 is kept as a roster member and counted (HANDOFF.md calls it a placeholder)
  - a side with more than 5 players in one round is truncated and counted

What this file does not do
  - It does not train anything and does not write any file (no tomes/, data/combined/ or models/).
  - It does not use the test split for the scaling window.
  - It does not emit seconds with no living player, because those have no rows in the source.
  - It does not check the unresolved day counts for 2026-10-09 and 2026-10-10 (HANDOFF.md).
  - It does not install packages.

Smoke test (one day, de_mirage):
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\model\\embed_round_state.py
"""

import importlib.util
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
POSITIONS_DIR = ROOT / "data" / "positions_1hz"
SNAPSHOT_PATH = ROOT / "data" / "combined" / "cache_snapshot_2026.parquet"
MANIFEST_PATH = ROOT / "data" / "combined" / "split_manifest.json"

SPLITS = ("train", "val", "test")
SIDE_T, SIDE_CT = 2, 3
SLOTS_PER_SIDE = 5
N_SLOTS = 2 * SLOTS_PER_SIDE  # slots 0-4 CT, 5-9 T
FEATURES = ("alive", "x", "y", "z", "sin_yaw", "cos_yaw")
N_FEAT = len(FEATURES)
SENTINEL_PLAYER_ID = 65535
COORD_SENTINEL_ABS = 1.0e4  # a coordinate this large or larger is a recording sentinel (for example z = -16384)
LATENT_DIM = 8
POSITION_COLS = ["match_key", "round", "second", "player_id", "side", "x", "y", "z", "yaw"]

# Looked up without importing torch (importing it is slow and not needed here).
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@dataclass
class MapDataset:
    map_name: str
    X: np.ndarray        # (N, 10, 6) float32
    split: np.ndarray    # (N,) str: "train", "val" or "test"
    match_key: np.ndarray
    round: np.ndarray    # (N,) int32
    second: np.ndarray   # (N,) int32
    bounds: dict         # scaling window used for x, y, z
    stats: dict


@dataclass
class PCABaseline:
    mean: np.ndarray                      # (60,)
    components: np.ndarray                # (k, 60)
    explained_variance_ratio: np.ndarray  # (k,)


# ----------------------------------------------------------------- inputs and guards

def load_assign(manifest_path=MANIFEST_PATH) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        assign = json.load(f)["assign"]
    bad = sorted({v for v in assign.values() if v not in SPLITS})
    if bad:
        raise ValueError(f"unknown split labels in manifest: {bad}")
    return assign


def load_match_maps(snapshot_path=SNAPSHOT_PATH) -> dict:
    pairs = (
        pl.scan_parquet(str(snapshot_path))
        .select(["match_key", "map_name"])
        .unique()
        .collect()
    )
    multi = pairs.group_by("match_key").len().filter(pl.col("len") > 1)
    if multi.height:
        raise ValueError(f"{multi.height} match_keys have more than one map_name")
    return dict(zip(pairs["match_key"].to_list(), pairs["map_name"].to_list()))


def split_members(assign: Mapping[str, str]) -> dict:
    members = {s: set() for s in SPLITS}
    for key, split in assign.items():
        if split not in members:
            raise ValueError(f"unknown split {split!r} for {key}")
        members[split].add(key)
    return members


def assert_no_split_overlap(members: Mapping[str, Iterable[str]]) -> None:
    owner = {}
    for split, keys in members.items():
        for key in keys:
            prev = owner.setdefault(key, split)
            if prev != split:
                raise ValueError(f"match_key in two splits: {key} ({prev} and {split})")


def _defaults(assign, match_maps):
    if assign is None:
        assign = load_assign()
    if match_maps is None:
        match_maps = load_match_maps()
    assert_no_split_overlap(split_members(assign))
    return assign, match_maps


def _as_paths(day_files) -> list:
    if isinstance(day_files, (str, Path)):
        day_files = [day_files]
    paths = [str(Path(p)) for p in day_files]
    if not paths:
        raise ValueError("day_files is empty")
    return paths


# ------------------------------------------------------------------- dataset builder

def _bad_coordinate_expr():
    """True for recording gaps: the exact origin, a coordinate of magnitude COORD_SENTINEL_ABS or more, or a non-finite value."""
    origin = (pl.col("x") == 0) & (pl.col("y") == 0) & (pl.col("z") == 0)
    big = (
        (pl.col("x").abs() >= COORD_SENTINEL_ABS)
        | (pl.col("y").abs() >= COORD_SENTINEL_ABS)
        | (pl.col("z").abs() >= COORD_SENTINEL_ABS)
    )
    not_finite = ~(pl.col("x").is_finite() & pl.col("y").is_finite() & pl.col("z").is_finite())
    return origin | big | not_finite


def _load_map_rows(map_name, paths, assign, match_maps):
    """Rows of one map, with slot (0-9) and split label. Returns (frame, stats)."""
    map_keys = sorted(k for k, m in match_maps.items() if m == map_name)
    if not map_keys:
        raise ValueError(f"no match_key has map_name={map_name!r}")

    known = list(match_maps)
    unmapped = sum(
        pl.scan_parquet(p)
        .select("match_key")
        .filter(~pl.col("match_key").is_in(known))
        .select(pl.len())
        .collect()
        .item()
        for p in paths
    )
    frames = [
        pl.scan_parquet(p)
        .select(POSITION_COLS)
        .filter(pl.col("match_key").is_in(map_keys))
        .collect()
        for p in paths
    ]
    df = pl.concat(frames)
    stats = {"position_rows_in_map": df.height, "position_rows_unmapped_match_key": unmapped}

    n_sentinel = df.filter(pl.col("player_id") == SENTINEL_PLAYER_ID).height
    df = df.filter(pl.col("player_id") != SENTINEL_PLAYER_ID)
    stats["rows_dropped_sentinel_pid"] = n_sentinel
    stats["rows_player_id_0_kept"] = df.filter(pl.col("player_id") == 0).height

    bad_xyz = _bad_coordinate_expr()
    stats["rows_dropped_bad_coordinates"] = df.filter(bad_xyz).height
    df = df.filter(~bad_xyz)

    if df.filter(pl.col("side").is_in([SIDE_T, SIDE_CT])).height != df.height:
        raise ValueError("side values other than 2 (T) and 3 (CT) found")
    n_dup = df.height - df.select(["match_key", "round", "second", "player_id"]).unique().height
    if n_dup:
        raise ValueError(f"{n_dup} duplicate (match_key, round, second, player_id) rows")

    # 0-based rank of player_id within (match, round, side), ascending by player_id.
    df = df.with_columns(
        (pl.col("player_id").rank("dense").over(["match_key", "round", "side"]) - 1)
        .cast(pl.Int32)
        .alias("rank0")
    )
    stats["rows_dropped_roster_overflow"] = df.filter(pl.col("rank0") >= SLOTS_PER_SIDE).height
    df = df.filter(pl.col("rank0") < SLOTS_PER_SIDE).with_columns(
        (pl.when(pl.col("side") == SIDE_CT).then(0).otherwise(SLOTS_PER_SIDE) + pl.col("rank0"))
        .cast(pl.Int32)
        .alias("slot")
    )

    labelled = {k: assign[k] for k in map_keys if k in assign}
    assert_no_split_overlap(split_members(labelled))
    split_df = pl.DataFrame(
        {"match_key": list(labelled), "split": list(labelled.values())},
        schema={"match_key": pl.String, "split": pl.String},
    )
    df = df.join(split_df, on="match_key", how="left")
    stats["rows_dropped_unlabelled"] = df.filter(pl.col("split").is_null()).height
    df = df.filter(pl.col("split").is_not_null())
    if df.height == 0:
        raise ValueError(f"no labelled rows for {map_name!r} in the given day files")
    return df, stats


def _fit_bounds(df: pl.DataFrame) -> dict:
    tr = df.filter(pl.col("split") == "train")
    if tr.height == 0:
        raise ValueError("no train rows for this map in the given day files; cannot fit a window")
    xy_min = min(float(tr["x"].min()), float(tr["y"].min()))
    xy_max = max(float(tr["x"].max()), float(tr["y"].max()))
    z_min = float(tr["z"].min())
    z_max = float(tr["z"].max())
    if not (xy_max > xy_min and z_max > z_min):
        raise ValueError("degenerate scaling window")
    return {
        "xy_min": xy_min,
        "xy_max": xy_max,
        "z_min": z_min,
        "z_max": z_max,
        "n_train_rows": tr.height,
    }


def _pack(df: pl.DataFrame, bounds: dict):
    key_df = (
        df.select(["match_key", "round", "second", "split"])
        .unique(subset=["match_key", "round", "second"], keep="first", maintain_order=True)
        .sort(["match_key", "round", "second"])
        .with_row_index("row")
    )
    if key_df.group_by("match_key").agg(pl.col("split").n_unique().alias("n")).filter(pl.col("n") > 1).height:
        raise ValueError("a match_key has rows with two split labels")

    spans = key_df.group_by(["match_key", "round"]).agg(
        pl.col("second").min().alias("lo"),
        pl.col("second").max().alias("hi"),
        pl.len().alias("n"),
    )
    rounds_gapped = spans.filter((pl.col("lo") != 0) | (pl.col("hi") - pl.col("lo") + 1 != pl.col("n"))).height
    if rounds_gapped:
        warnings.warn(f"{rounds_gapped} rounds do not run contiguously from second 0; kept, see stats", stacklevel=2)

    joined = df.join(
        key_df.select(["match_key", "round", "second", "row"]),
        on=["match_key", "round", "second"],
        how="left",
    )
    if joined["row"].null_count():
        raise ValueError("row index join failed")
    rows = joined["row"].to_numpy().astype(np.int64)
    slots = joined["slot"].to_numpy().astype(np.int64)

    xy_span = bounds["xy_max"] - bounds["xy_min"]
    z_span = bounds["z_max"] - bounds["z_min"]
    xs = (joined["x"].to_numpy().astype(np.float64) - bounds["xy_min"]) / xy_span
    ys = (joined["y"].to_numpy().astype(np.float64) - bounds["xy_min"]) / xy_span
    zs = (joined["z"].to_numpy().astype(np.float64) - bounds["z_min"]) / z_span
    yaw = np.deg2rad(joined["yaw"].to_numpy().astype(np.float64))
    feats = np.stack([np.ones_like(xs), xs, ys, zs, np.sin(yaw), np.cos(yaw)], axis=1)
    if not np.isfinite(feats).all():
        raise ValueError("non-finite x, y, z or yaw in the position rows")

    X = np.zeros((key_df.height, N_SLOTS, N_FEAT), dtype=np.float32)
    X[rows, slots] = feats.astype(np.float32)
    if int(X[:, :, 0].sum()) != joined.height:
        raise ValueError("two players share one slot in the same second")

    split_row = joined["split"].to_numpy().astype(str)
    coords = feats[:, 1:4]
    out_of_window = {
        s: float(np.mean((coords[split_row == s] < 0) | (coords[split_row == s] > 1)))
        for s in SPLITS
        if (split_row == s).any()
    }
    split_arr = key_df["split"].to_numpy().astype(str)
    stats = {
        "n_rows": key_df.height,
        "rounds_gapped": rounds_gapped,
        "rows_per_split": {s: int((split_arr == s).sum()) for s in SPLITS},
        "matches_per_split": {
            s: key_df.filter(pl.col("split") == s)["match_key"].n_unique() for s in SPLITS
        },
        "frac_alive_slots": float(X[:, :, 0].mean()),
        "frac_alive_coords_outside_window": out_of_window,
    }
    return key_df, X, stats


def fit_map_bounds(map_name, day_files, *, assign=None, match_maps=None) -> dict:
    """Scaling window from the TRAIN rows of one map across day_files.

    Fit it once on the training days and pass the result as bounds= to every
    build_map_dataset call, so all days and splits share one window.
    """
    assign, match_maps = _defaults(assign, match_maps)
    df, _ = _load_map_rows(map_name, _as_paths(day_files), assign, match_maps)
    return _fit_bounds(df)


def build_map_dataset(map_name, day_files, *, bounds=None, assign=None, match_maps=None) -> MapDataset:
    """Fixed 10-slot per-second arrays for one map.

    Slots 0-4 are CT and 5-9 are T. Each side is ordered by player_id within the round.
    Channels are alive, x, y, z, sin(yaw), cos(yaw). Missing or dead slots are zeros.
    If bounds is None the window is fitted on these day_files only, which is only correct
    when day_files are all the training data. Otherwise pass fit_map_bounds(...) output.
    """
    paths = _as_paths(day_files)
    assign, match_maps = _defaults(assign, match_maps)
    df, stats = _load_map_rows(map_name, paths, assign, match_maps)
    if bounds is None:
        warnings.warn(
            "scaling window fitted from these day_files only; pass bounds=fit_map_bounds(...) "
            "from all training days so every day and split uses one window",
            stacklevel=2,
        )
        bounds = _fit_bounds(df)
    key_df, X, pack_stats = _pack(df, bounds)
    stats.update(pack_stats)
    return MapDataset(
        map_name=map_name,
        X=X,
        split=key_df["split"].to_numpy().astype(str),
        match_key=key_df["match_key"].to_numpy().astype(str),
        round=key_df["round"].cast(pl.Int32).to_numpy(),
        second=key_df["second"].cast(pl.Int32).to_numpy(),
        bounds=bounds,
        stats=stats,
    )


# ----------------------------------------------------------------------- embeddings

def fit_pca_baseline(X_train: np.ndarray, n_components: int = LATENT_DIM) -> PCABaseline:
    """PCA on the flattened (10 * 6) arrays. Fit on TRAIN rows only.

    This is an unsupervised linear projection with dead slots entered as zeros. It is
    not permutation-invariant over players, so it is a baseline only.
    """
    flat = np.asarray(X_train, dtype=np.float64).reshape(len(X_train), -1)
    mean = flat.mean(axis=0)
    centred = flat - mean
    cov = centred.T @ centred / max(len(flat) - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1][:n_components]
    total = float(eigval.sum())
    ratio = eigval[order] / total if total > 0 else np.zeros(len(order))
    return PCABaseline(
        mean=mean,
        components=eigvec[:, order].T.copy(),
        explained_variance_ratio=ratio,
    )


def transform_pca_baseline(model: PCABaseline, X: np.ndarray) -> np.ndarray:
    flat = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    return ((flat - model.mean) @ model.components.T).astype(np.float32)


def embed_deepsets(X: np.ndarray, *, latent_dim: int = LATENT_DIM) -> np.ndarray:
    """SIGNATURE ONLY. Not implemented, because torch is not installed in .pixi/envs/default.

    Intended design (latent_dim = 8, one encoder shared by all rows):
      1. phi: a per-slot MLP R^6 -> R^32, with the same weights for all 10 slots.
      2. Mask: multiply phi(slot) by alive, so dead and missing slots contribute nothing.
      3. Pool: sum over each team's five slots (0-4 CT, 5-9 T), giving two 32-d vectors.
         A sum does not depend on the order of players within a team.
      4. rho: concatenate the two team sums (64-d) and apply an MLP to R^latent_dim.
    Returns an (N, latent_dim) float32 array. Training is out of scope for this file.
    """
    raise NotImplementedError(
        "torch is not installed in .pixi/envs/default; the DeepSets encoder is design-only"
    )


# ----------------------------------------------------------------------- smoke test

def _smoke() -> None:
    t0 = time.perf_counter()
    day = POSITIONS_DIR / "2026-10-01.parquet"
    assign, match_maps = _defaults(None, None)
    print(f"torch available: {TORCH_AVAILABLE}")
    print(f"manifest entries: {len(assign)}; split overlap guard: passed")

    bounds = fit_map_bounds("de_mirage", [day], assign=assign, match_maps=match_maps)
    ds = build_map_dataset("de_mirage", [day], bounds=bounds, assign=assign, match_maps=match_maps)
    assert ds.X.shape[1:] == (N_SLOTS, N_FEAT)
    print(f"map: {ds.map_name}  day: {day.name}  matches: {len(set(ds.match_key.tolist()))}")
    print(f"X shape: {ds.X.shape}  dtype: {ds.X.dtype}")
    print("rows per split: " + ", ".join(f"{s}={int((ds.split == s).sum())}" for s in SPLITS))
    print(f"fraction of slots alive: {ds.X[:, :, 0].mean():.4f}")
    print(f"scaling window (smoke only, fitted on this one day): {ds.bounds}")
    print(f"stats: {ds.stats}")
    print(f"elapsed: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    _smoke()
