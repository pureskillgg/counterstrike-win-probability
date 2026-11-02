"""Paired comparison on the same de_mirage test seconds: the 57-input baseline against a temporal model.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\model\\compare_paired_de_mirage.py --temporal data\\temporal\\de_mirage_gat_loadout_gru_pred.npz

The baseline (data/combined/export_winprob_v6_drop_heavy_test_predictions.parquet) is keyed by tick; the temporal predictions are
keyed by second. Ticks map to seconds through the positions files (min tick per match, round and second). Rows are joined on
(match_key, round, second), and only seconds present in both are compared. The difference in log loss is paired per second, and a
1,000-resample match-cluster bootstrap (seed 0) gives its interval. Writes data/temporal/paired_<tag>_vs_57_de_mirage.json.
load_paired() is shared with score_paired_metrics.py so both scripts score the same seconds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
COMBINED = ROOT / "data" / "combined"
POSITIONS = ROOT / "data" / "positions_1hz"
OUT = ROOT / "data" / "temporal"
MAP = "de_mirage"


def row_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_paired(temporal_path):
    """Baseline and temporal predictions joined on the same test seconds. Returns (paired frame, model tag)."""
    tag = Path(temporal_path).stem.replace("_pred", "")

    snap = pl.read_parquet(COMBINED / "cache_snapshot_2026.parquet", columns=["match_key", "map_name"]).unique("match_key")
    de_keys = snap.filter(pl.col("map_name") == MAP)["match_key"].to_list()
    rounds = (pl.read_parquet(COMBINED / "rounds.parquet", columns=["match_key", "round", "final"])
              .with_columns(pl.col("round").cast(pl.Int64)).unique(["match_key", "round"], keep="first"))

    # tick -> second: min tick per (match, round, second), from the positions files
    tick_parts = []
    for f in sorted(POSITIONS.glob("*.parquet")):
        part = (pl.read_parquet(f, columns=["match_key", "round", "second", "tick"])
                .filter(pl.col("match_key").is_in(de_keys))
                .group_by(["match_key", "round", "second"]).agg(pl.col("tick").min()))
        tick_parts.append(part)
    ticks = pl.concat(tick_parts).with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))

    export = pl.read_parquet(COMBINED / "export_winprob_v6_drop_heavy_test_predictions.parquet")
    base = (export.filter(pl.col("match_key").is_in(de_keys))
            .join(ticks, on=["match_key", "round", "tick"], how="inner")
            .select(["match_key", "round", "second", pl.col("p").alias("p_base")]))

    npz = np.load(temporal_path)
    keys = npz["match_keys"]
    test = npz["split"] == 2
    temp = pl.DataFrame({
        "match_key": keys[npz["match_idx"][test]],
        "round": npz["round"][test].astype(np.int64),
        "second": npz["second"][test].astype(np.int64),
        "p_temp": npz["p"][test].astype(np.float64),
    })

    paired = (base.join(temp, on=["match_key", "round", "second"], how="inner")
              .join(rounds, on=["match_key", "round"], how="inner")
              .with_columns((pl.col("final") == 3).cast(pl.Float64).alias("y")))
    return paired, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal", required=True, help="npz from train_temporal_round_model.py or train_thgnn_lite.py (p, split, match_idx, round, second, match_keys)")
    a = ap.parse_args()
    paired, tag = load_paired(a.temporal)
    y = paired["y"].to_numpy()
    pb = paired["p_base"].to_numpy()
    pt = paired["p_temp"].to_numpy()
    lb, lt = row_loss(y, pb), row_loss(y, pt)
    diff = lt - lb  # positive means the temporal model is worse

    match = paired["match_key"].to_numpy()
    uniq, inv = np.unique(match, return_inverse=True)
    sums = np.zeros(uniq.size)
    counts = np.zeros(uniq.size)
    np.add.at(sums, inv, diff)
    np.add.at(counts, inv, 1)
    rng = np.random.default_rng(0)
    boot = np.empty(1000)
    for b in range(1000):
        pick = rng.integers(0, uniq.size, uniq.size)
        boot[b] = sums[pick].sum() / counts[pick].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])

    result = {
        "map": MAP,
        "temporal_model": tag,
        "baseline": "winprob_v6_drop_heavy (57 inputs)",
        "paired_seconds": int(paired.height),
        "paired_matches": int(uniq.size),
        "log_loss_baseline": float(lb.mean()),
        "log_loss_temporal": float(lt.mean()),
        "difference_temporal_minus_baseline": float(diff.mean()),
        "ci95_match_bootstrap": [float(lo), float(hi)],
        "bootstrap_resamples": 1000,
        "seed": 0,
        "reading": "positive difference: the temporal model is worse on the same seconds",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"paired_{tag}_vs_57_{MAP}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
