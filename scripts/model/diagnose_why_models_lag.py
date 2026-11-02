"""Why do the temporal models trail the 57-input baseline? Controlled ablations on the frozen split, de_mirage test.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 -u scripts\\model\\diagnose_why_models_lag.py

Same XGBoost settings as the control (exp_loadout_ablation.py: depth 4, eta 0.05, min child weight 20, subsample 0.8, early
stopping 100 on validation seconds, monotone alive counts). Runs on CPU so the GPU stays free while tomes are built.
Frozen split: data/combined/split_manifest.json. Scored on de_mirage test matches only, with W.evaluate (round-weighted log loss,
match-cluster bootstrap). Each experiment answers one question:

  C0 control         59 inputs, all maps, train on all-map train          -> the reference
  A1 single map      59 inputs minus map one-hots, de_mirage train only  -> does restricting to one map help?
  S1 state only      6 grid inputs, all maps                              -> how much the 53 snapshot inputs add
  S2 state only, 1m  6 grid inputs, de_mirage train only                  -> single-map state-only
  L1 lagged change   59 inputs plus 5 and 20 second changes, all maps     -> does temporal context help the strong model?
  L2 lagged, 1 map   as L1, de_mirage train only                          -> single-map with context

Writes data/diagnostics/why_models_lag.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

import wp_common as W

OUT = os.path.join(W.ROOT, "data", "diagnostics", "why_models_lag.json")
MAP = "de_mirage"
LAG_BASE = ["alive_ct", "alive_t", "st_hp_ct", "st_hp_t", "st_equip_ct", "st_equip_t", "st_armor_sum_ct", "st_armor_sum_t",
            "st_rifle_ct", "st_rifle_t"]
LAGS = [5, 20]


def load():
    meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    feats59 = meta
    need = ["match_key", "round", "second", "label_ct_win", "map_name", "era"] + feats59
    G = pl.read_parquet(os.path.join(W.ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=need)
    G = W.drop_roster_defect_rounds(G)
    man = W.load_manifest(os.path.join(W.ROOT, "data", "combined", "split_manifest.json"))
    assign = man["assign"]
    keys = G["match_key"].unique().to_list()
    unassigned = [k for k in keys if k not in assign]
    A = pl.DataFrame({"match_key": list(assign.keys()), "split": list(assign.values())})
    G = G.join(A, on="match_key", how="inner")
    G = G.filter(pl.col("label_ct_win").is_not_null())
    G = G.sort(["match_key", "round", "second"])
    G = G.with_columns((1.0 / pl.len().over(["match_key", "round"])).cast(pl.Float64).alias("w_round"))  # each round weighs 1
    lag_cols = []
    for k in LAGS:
        for c in LAG_BASE:
            name = f"d{k}_{c}"
            G = G.with_columns((pl.col(c) - pl.col(c).shift(k).over(["match_key", "round"]).fill_null(pl.col(c))).alias(name))
            lag_cols.append(name)
    print(f"rows {G.height:,}; keys without a manifest entry {len(unassigned)}", flush=True)
    return G, feats59, lag_cols


TRAIN_STRIDE = 2  # every second train row only, to fit the lagged runs in memory; evaluation always uses every test second


def save(results):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"map": MAP, "split": "frozen manifest", "train_stride": TRAIN_STRIDE, "results": results}, f, indent=1)


def run(G, name, feats, train_map_only, mono_feats, results):
    t0 = time.time()
    base = G.filter(pl.col("map_name") == MAP) if train_map_only else G
    tr = base.filter((pl.col("split") == "train") & (pl.col("second") % TRAIN_STRIDE == 0))
    va = base.filter(pl.col("split") == "val")
    te = G.filter((pl.col("split") == "test") & (pl.col("map_name") == MAP))
    if train_map_only:
        te = te  # test is always de_mirage
    Xtr = tr.select(feats).to_numpy().astype("float32")
    ytr = tr["label_ct_win"].to_numpy().astype("int64")
    Xva = va.select(feats).to_numpy().astype("float32")
    yva = va["label_ct_win"].to_numpy().astype("int64")
    mono = W.mono_for(mono_feats) if mono_feats else None
    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono=mono, seed=0, device="cuda")
    pte = W.predict(m, best, te.select(feats).to_numpy().astype("float32"))
    yte = te["label_ct_win"].to_numpy().astype("int64")
    ev = W.evaluate(te, yte, pte)
    row = {"experiment": name, "features": len(feats), "train_rows": tr.height, "train_matches": tr["match_key"].n_unique(),
           "test_rows": te.height, "test_matches": te["match_key"].n_unique(), "best_iteration": int(best),
           "test": ev, "seconds": round(time.time() - t0, 1)}
    results.append(row)
    print(f"{name}: test log loss {ev['logloss']:.5f}  train matches {row['train_matches']:,}  "
          f"best iter {best}  ({row['seconds']}s)", flush=True)
    save(results)  # written after every run, so a crash keeps the finished runs
    return row


def main():
    G, feats59, lag_cols = load()
    map_feats = [f for f in feats59 if f.startswith("map_is_")]
    feats_nomap = [f for f in feats59 if f not in map_feats]
    state6 = list(W.F6)
    results = []
    run(G, "C0 control: 59 inputs, all maps", feats59, False, feats59, results)
    run(G, "A1 single map: 59 minus map one-hots, de_mirage train", feats_nomap, True, feats_nomap, results)
    run(G, "S1 state only: 6 grid inputs, all maps", state6, False, state6, results)
    run(G, "S2 state only: 6 grid inputs, de_mirage train", state6, True, state6, results)
    lagged = feats59 + lag_cols
    run(G, "L1 lagged change: 59 plus lag changes, all maps", lagged, False, lagged, results)
    run(G, "L2 lagged change: 59 plus lag changes, de_mirage train", [f for f in lagged if f not in map_feats], True,
        [f for f in lagged if f not in map_feats], results)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"map": MAP, "split": "frozen manifest", "results": results}, f, indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
