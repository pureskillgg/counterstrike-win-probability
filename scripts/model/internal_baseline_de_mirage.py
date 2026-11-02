"""Internal baseline on the de_mirage test matches: the 57-input XGBoost (winprob_v6_drop_heavy) per-second predictions.

  G:\pureskill\csds-cs2\.pixi\envs\default\python.exe -I -X utf8 scripts\model\internal_baseline_de_mirage.py

Reads data/combined/export_winprob_v6_drop_heavy_test_predictions.parquet (per-second p = P(CT wins), frozen test split),
data/combined/cache_snapshot_2026.parquet (map_name), data/combined/split_manifest.json (split), data/combined/rounds.parquet
(final: 2 = T, 3 = CT). Scores log loss on the rows of de_mirage test matches, with a 1,000-resample match-cluster bootstrap
(seed 0). Writes data/combined/internal_baseline_de_mirage_test.json. Nothing else is written.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
COMBINED = ROOT / "data" / "combined"
MAP = "de_mirage"
N_BOOT = 1000


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    export = pl.read_parquet(COMBINED / "export_winprob_v6_drop_heavy_test_predictions.parquet")
    snap = (pl.read_parquet(COMBINED / "cache_snapshot_2026.parquet", columns=["match_key", "map_name"])
            .unique(["match_key"], keep="first"))
    manifest = json.loads((COMBINED / "split_manifest.json").read_text(encoding="utf-8"))["assign"]
    rounds = (pl.read_parquet(COMBINED / "rounds.parquet", columns=["match_key", "round", "final"])
              .with_columns(pl.col("round").cast(pl.Int64))
              .unique(["match_key", "round"], keep="first"))

    test_keys = {k for k, v in manifest.items() if v == "test"}
    df = (export.join(snap, on="match_key", how="left")
          .filter((pl.col("map_name") == MAP) & pl.col("match_key").is_in(list(test_keys)))
          .join(rounds, on=["match_key", "round"], how="left"))
    n_missing = df.filter(pl.col("final").is_null()).height
    df = df.filter(pl.col("final").is_not_null()).with_columns(
        (pl.col("final") == 3).cast(pl.Float64).alias("y"),  # 1 when CT wins
    )
    y = df["y"].to_numpy()
    p = df["p"].to_numpy()
    match = df["match_key"].to_numpy()
    print(f"de_mirage test rows {df.height}, matches {len(set(match.tolist()))}, rows without a winner {n_missing}")
    print(f"mean p {p.mean():.4f} vs CT win rate {y.mean():.4f}")
    print(f"mean p when CT wins {p[y == 1].mean():.4f}, when T wins {p[y == 0].mean():.4f} (p must be higher when CT wins)")

    point = log_loss(y, p)

    # match-cluster bootstrap: resample matches with replacement
    uniq, inverse = np.unique(match, return_inverse=True)
    rng = np.random.default_rng(0)
    sums = np.zeros(uniq.size)
    counts = np.zeros(uniq.size)
    row_loss = -(y * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1 - 1e-6)))
    np.add.at(sums, inverse, row_loss)
    np.add.at(counts, inverse, 1)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, uniq.size, uniq.size)
        boot[b] = sums[pick].sum() / counts[pick].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    result = {
        "map": MAP,
        "split": "test (frozen manifest)",
        "model": "winprob_v6_drop_heavy (57 inputs, XGBoost), per-second p = P(CT wins)",
        "rows": int(df.height),
        "matches": int(uniq.size),
        "log_loss": point,
        "ci95_match_bootstrap": [float(lo), float(hi)],
        "bootstrap_resamples": N_BOOT,
        "seed": 0,
    }
    (COMBINED / "internal_baseline_de_mirage_test.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
