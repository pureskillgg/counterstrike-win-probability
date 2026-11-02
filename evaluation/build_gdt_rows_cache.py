"""Build intern-grader rows from the cache alone, for the test-assigned matches of a window.

These are the matches the manifest marks as test that are not in the frozen benchmark, so no model here trained on them. The
intern's protocol window is upload days Oct 14 to 25, 2026. Every second of every qualifying round is a row, as the README asks.

Predictions: v4 and v6 are recomputed from the cache. Extra models are given as --model name=path.json (their features are v4's
59 inputs, as in the retrain scripts).

    python evaluation/build_gdt_rows_cache.py --cache <parquet> --d0 2026/10/14 --d1 2026/10/25 \
        --model p_exp=<model.json> --out <rows.parquet>
"""
import argparse
import ast
import json
import os

import numpy as np
import polars as pl
import xgboost as xgb

ROOT = "G:/pureskill/csds-cs2"
SCR = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
MAPS = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]
KEY = ["match_key", "round", "second"]


def feats_of(meta):
    f = meta["features"]
    return ast.literal_eval(f) if isinstance(f, str) else list(f)


def predict(model_path, meta, feats, frame):
    bst = xgb.Booster()
    bst.load_model(model_path)
    X = frame.select(feats).to_numpy().astype(np.float32)
    return bst.predict(xgb.DMatrix(X), iteration_range=(0, int(meta["best_iteration"]) + 1))


def main():
    ap = argparse.ArgumentParser(description="Intern-grader rows from the cache for the test-assigned matches of a window")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--d0", default="2026/10/14")
    ap.add_argument("--d1", default="2026/10/25")
    ap.add_argument("--model", action="append", default=[], help="name=path.json for an extra model (v4 features)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    frozen = set(json.load(open(SCR + "/frozen_test_keys.json")))
    man = json.load(open(ROOT + "/data/combined/split_manifest.json"))["assign"]
    m4 = json.load(open(ROOT + "/models/winprob_v4.meta.json"))
    m6 = json.load(open(ROOT + "/models/winprob_v6_drop_heavy.meta.json"))
    F4, F6 = feats_of(m4), feats_of(m6)

    # 1. the test-assigned matches in the window, not frozen, on the seven maps, in scope, max_rounds 24
    allkeys = pl.scan_parquet(a.cache).select(["match_key", "map_name"]).unique().collect()
    allkeys = allkeys.with_columns(pl.col("match_key").str.extract(r"csds/(\d{4}/\d{2}/\d{2})", 1).alias("day"))
    sel = allkeys.filter((pl.col("day") >= a.d0) & (pl.col("day") <= a.d1) & pl.col("map_name").is_in(MAPS))
    sel = sel.filter(pl.col("match_key").map_elements(lambda k: man.get(k) == "test" and k not in frozen, return_dtype=pl.Boolean))
    mt = pl.read_parquet(ROOT + "/data/combined/matches.parquet", columns=["match_key", "in_scope", "max_rounds_final"])
    sel = sel.join(mt.filter((pl.col("in_scope") == True) & (pl.col("max_rounds_final") == 24)).select("match_key"),  # noqa: E712
                   on="match_key", how="inner")
    keys = sel["match_key"].to_list()
    print("test-assigned matches in window, in scope:", len(keys), flush=True)

    # 2. every second of those matches; 5v5 round starts only
    need = sorted(set(F4) | set(F6) | {"match_key", "round", "second", "alive_ct", "alive_t", "st_equip_ct", "st_equip_t",
                                       "bomb_planted", "seconds_since_plant", "label_ct_win"})
    df = pl.scan_parquet(a.cache).filter(pl.col("match_key").is_in(keys)).select(need).collect()
    starts = df.filter((pl.col("second") == 0) & (pl.col("alive_ct") == 5) & (pl.col("alive_t") == 5)).select(["match_key", "round"])
    df = df.join(starts, on=["match_key", "round"], how="inner")
    print("rows", df.height, "rounds", starts.height, flush=True)

    # 3. predictions
    df = df.with_columns(pl.Series("p_v4", predict(ROOT + "/models/winprob_v4.json", m4, F4, df)),
                         pl.Series("p_v6", predict(ROOT + "/models/winprob_v6_drop_heavy.json", m6, F6, df)))
    extra = []
    for spec in a.model:
        name, path = spec.split("=", 1)
        df = df.with_columns(pl.Series(name, predict(path, m4, F4, df)))
        extra.append(name)

    # 4. the grader's columns; since_plant is 0 before the plant
    out = df.select([
        pl.col("match_key").alias("match"), "round", "second", "alive_ct", "alive_t",
        pl.col("st_equip_ct").alias("equip_ct"), pl.col("st_equip_t").alias("equip_t"),
        pl.col("bomb_planted").cast(pl.Int64).alias("planted"),
        pl.when(pl.col("bomb_planted") == 1).then(pl.col("seconds_since_plant")).otherwise(0.0).alias("since_plant"),
        pl.col("label_ct_win").cast(pl.Int64).alias("y"),
        "p_v6", "p_v4", *extra,
    ])
    out.write_parquet(a.out)
    print("WROTE", a.out, "rows", out.height, "matches", out["match"].n_unique(), flush=True)


if __name__ == "__main__":
    main()
