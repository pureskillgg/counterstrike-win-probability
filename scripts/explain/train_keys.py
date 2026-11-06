"""Rebuilds the row keys of the sprint's design matrix, training rows included, without rewriting X.

Why: sprint_prepare.py saved keys only for the validation rows (val_keys.parquet). A GDT-matched training needs each training
second's match and round, so this recomputes the same rows in the same order: the same split (wp_common.make_split, seed 0),
the same early-stopping set, and the same inner join to the position table as sprint_prepare.py, keeping only the key columns.

Checks, before anything is written:
  - the fit, early-stop and validation counts equal matrix_trainctr/meta.json (n_fit, n_es, n_val);
  - the validation rows, in order, equal val_keys.parquet on match, round, second and label.
If either fails, nothing is written.

Output: matrix_trainctr/train_keys.parquet, one row per matrix row, in matrix order, with a column "part" (fit, es or val).

Run with the pixi Python from the repo root:
    G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe scripts\\explain\\train_keys.py
"""

import json
import os
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "sprint"))
sys.path.insert(0, str(REPO / "scripts" / "model"))
import sprint_worker as SW  # noqa: E402
import wp_common as W  # noqa: E402

MATRIX = Path(os.environ.get("CS2_EXPLAIN_MATRIX", "C:/Users/<user>/sprint/matrix_trainctr"))


def main():
    meta = json.load(open(MATRIX / "meta.json"))
    cache_meta = pl.scan_parquet(SW.TRAIN_CACHE).select(["match_key", "era", "match_date"]).unique(subset=["match_key"]).collect()
    sp = W.make_split("random", cache_meta, seed=0)
    es = SW.early_stop_set(sp["train"])
    fit = sp["train"] - es
    val = sp["val"]
    assert not ((fit | es | val) & sp["test"]), "split sets overlap the test split"
    keep = sorted(fit | es | val)

    pos = pl.scan_parquet(os.path.join(SW.ROOT, "data", "combined", SW.POS_NAME)).select(SW.POS_KEY)
    G = (pl.scan_parquet(SW.TRAIN_CACHE)
         .filter(pl.col("match_key").is_in(keep))
         .select(SW.POS_KEY + ["label_ct_win", "second"])
         .join(pos, on=SW.POS_KEY, how="inner")
         .collect())

    part = pl.Series("part", [("fit" if m in fit else "es" if m in es else "val") for m in G["match_key"].to_list()])
    keys = G.with_columns(part)
    counts = {p: int((keys["part"] == p).sum()) for p in ("fit", "es", "val")}
    print("counts", counts, "meta", {k: meta[k] for k in ("n_fit", "n_es", "n_val")}, flush=True)
    assert counts["fit"] == meta["n_fit"] and counts["es"] == meta["n_es"] and counts["val"] == meta["n_val"], \
        "row counts differ from meta.json"
    assert keys.height == meta["rows"], "total rows differ from meta.json"

    # matrix order is fit, early stop, validation: each part in the order of the join
    ordered = pl.concat([keys.filter(pl.col("part") == p) for p in ("fit", "es", "val")])
    val_rows = ordered.filter(pl.col("part") == "val").select(["match_key", "round", "second", "label_ct_win"])
    saved = pl.read_parquet(MATRIX / "val_keys.parquet").select(["match_key", "round", "second", "label_ct_win"])
    same = val_rows.height == saved.height and val_rows.equals(saved)
    print("validation keys equal val_keys.parquet row for row:", same, flush=True)
    assert same, "rebuilt validation keys differ from val_keys.parquet; nothing written"

    out = MATRIX / "train_keys.parquet"
    ordered.write_parquet(out)
    print("wrote", out, ordered.height, "rows", flush=True)


if __name__ == "__main__":
    main()
