"""Writes the sprint's design matrix to disk, once, for the workers.

The matrix holds the training-cache rows that have a train-centre position row, ordered fit, early stop, validation. A worker then
reads two contiguous slices per fit instead of keeping the whole matrix in memory, which the 19 GB WSL limit needs (a fit's own copies
took the process over that limit). Holdout B and the frozen test are asserted out of the fit, stop and validation sets here, before any
row is written. The row rules are those of experiments/positions/positions_validation.py.

    CS2_ROOT=/mnt/g/pureskill/csds-cs2 python sprint_prepare.py --pos <cache_pos_2026_trainctr.parquet> \
        --frozen <frozen_test_keys.json> --out <matrix dir>
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import sprint_worker as SW  # noqa: E402
import wp_common as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Write the validation sprint design matrix")
    ap.add_argument("--pos", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.basename(a.pos) != SW.POS_NAME:
        raise SystemExit(f"--pos must be {SW.POS_NAME} (site centres from training matches only), not {os.path.basename(a.pos)}")

    names = list(pl.read_parquet_schema(SW.TRAIN_CACHE).keys())
    cache_base = [c for c in W.F6 if c in names] + [c for c in names if c.startswith("map_is_")] + [c for c in names if c.startswith("st_")]
    base = list(dict.fromkeys(cache_base))
    pos_cols = SW.GEO + SW.VIEW + SW.BLIND
    keep_cols = ["second"] if "second" in names else []
    sec = keep_cols[0] if keep_cols else "tick"

    meta = pl.scan_parquet(SW.TRAIN_CACHE).select(["match_key", "era", "match_date"]).unique(subset=["match_key"]).collect()
    sp = W.make_split("random", meta, seed=0)
    es = SW.early_stop_set(sp["train"])
    fit = sp["train"] - es
    val = sp["val"]
    assert not (val & sp["test"]) and not (fit & sp["test"]) and not (es & sp["test"]), "split sets overlap"
    holdout = set(json.load(open(SW.HOLDOUT_B_KEYS))["match_keys"])
    frozen = set(json.load(open(a.frozen)))
    assert not ((fit | es | val) & (holdout | frozen)), "holdout B or frozen test key entered a fit, stop set or score"
    print(f"fit matches {len(fit)}, early-stopping matches {len(es)}, validation matches {len(val)}; "
          "holdout B and frozen test not selected", flush=True)

    keep = sorted(fit | es | val)
    pos = pl.scan_parquet(a.pos).select(SW.POS_KEY + pos_cols).with_columns([pl.col(c).cast(pl.Float32) for c in pos_cols])
    G = (pl.scan_parquet(SW.TRAIN_CACHE)
         .filter(pl.col("match_key").is_in(keep))
         .select(SW.POS_KEY + ["label_ct_win"] + base + keep_cols)
         .with_columns([pl.col(c).cast(pl.Float32) for c in base])
         .join(pos, on=SW.POS_KEY, how="inner")
         .collect())
    feats_all = base + pos_cols
    masks = {"fit": G["match_key"].is_in(sorted(fit)).to_numpy(), "es": G["match_key"].is_in(sorted(es)).to_numpy(),
             "val": G["match_key"].is_in(sorted(val)).to_numpy()}
    counts = {k: int(v.sum()) for k, v in masks.items()}
    N = sum(counts.values())
    print(f"rows with positions {G.height:,}; fit {counts['fit']:,}, early stop {counts['es']:,}, validation {counts['val']:,}; "
          f"rss {SW.rss_mb()} MB", flush=True)

    os.makedirs(a.out, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(a.out, "X.npy"), mode="w+", dtype=np.float32, shape=(N, len(feats_all)))
    y = np.empty(N, dtype=np.int64)
    off = 0
    for name in ("fit", "es", "val"):
        sub = G.filter(pl.Series("m", masks[name])).select(feats_all + ["label_ct_win"])
        k = sub.height
        X[off:off + k] = sub.select(feats_all).to_numpy().astype(np.float32, copy=False)
        y[off:off + k] = sub["label_ct_win"].to_numpy().astype(np.int64)
        off += k
        del sub
    assert off == N, "row count changed while writing the matrix"
    X.flush()
    del X
    np.save(os.path.join(a.out, "y.npy"), y)
    G.filter(pl.Series("m", masks["val"])).select(["match_key", "round", sec, "label_ct_win"]).write_parquet(
        os.path.join(a.out, "val_keys.parquet"))
    SW.write_json(os.path.join(a.out, "meta.json"), {
        "pos": os.path.basename(a.pos), "pos_bytes": os.path.getsize(a.pos), "cache": os.path.basename(SW.TRAIN_CACHE),
        "cache_bytes": os.path.getsize(SW.TRAIN_CACHE), "frozen": os.path.basename(a.frozen), "feats_all": feats_all, "base": base,
        "n_fit": counts["fit"], "n_es": counts["es"], "n_val": counts["val"], "rows": N})
    print(f"wrote matrix {a.out}: {N:,} rows x {len(feats_all)} columns", flush=True)


if __name__ == "__main__":
    main()
