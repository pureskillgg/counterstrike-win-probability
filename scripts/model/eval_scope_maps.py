"""How does the final model do on 5v5 competitive matches of maps it was NOT trained for (vertigo, overpass, train, ...)?  (CPU)

  ...python.exe -I -X utf8 scripts\\model\\eval_scope_maps.py [--name winprob_v3] [--basic winprob_v3_basic]
Inputs: data/scopetest (make_scope_test_store.py, make_grid.py --store scopetest), data/status_1hz_scopetest (extract_status_1hz.py ...).
Snapshot model on the 2026 matches (map one-hots all zero: a map the model does not know), alive-count fallback on the 2024 matches. Reference = the same model on its own
frozen held-out matches of the seven known maps. Match-cluster 95% intervals. Writes data/combined/eval_scope_maps_<name>.json. The serving code flags all of these
as out_of_scope; this measures how good the number still is.
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import json
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import features_status as FS
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]


def booster(name):
    meta = json.load(open(os.path.join(W.ROOT, "models", f"{name}.meta.json"), encoding="utf-8"))
    b = xgb.Booster()
    b.load_model(os.path.join(W.ROOT, "models", f"{name}.json"))
    return b, meta["features"], int(meta["best_iteration"])


def predict(b, feats, best, G):
    X = G.select(feats).to_numpy().astype(np.float32)
    return b.predict(xgb.DMatrix(X, feature_names=feats), iteration_range=(0, best + 1))


def block(sub, y, p):
    r = W.evaluate(sub, y, p)
    return {"n_matches": r["n_matches"], "n_rows": r["n_rows"], "logloss": r["logloss"], "logloss_ci95": r["logloss_ci95"], "auc": r["auc"], "ece10": r["ece10"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="winprob_v3")
    ap.add_argument("--basic", default="winprob_v3_basic")
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", "scopetest")
    G = pl.read_parquet(os.path.join(d, "grid.parquet"))
    M = pl.read_parquet(os.path.join(d, "matches.parquet")).select("match_key", "match_date")
    G = G.join(M, on="match_key", how="left")
    out = {"name": a.name, "basic": a.basic}
    # ---- alive-count fallback on every out-of-pool match (both eras)
    bb, fb, ib = booster(a.basic)
    Gb = G.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Int8).alias(f"map_is_{m}") for m in MAPS])
    pb = predict(bb, fb, ib, Gb)
    yb = Gb["label_ct_win"].to_numpy().astype(np.int64)
    out["fallback_all"] = block(Gb, yb, pb)
    out["fallback_by_map"] = {}
    for mp in ("de_vertigo", "de_overpass"):
        m = (Gb["map_name"] == mp).to_numpy()
        if m.sum() > 1000:
            out["fallback_by_map"][mp] = block(Gb.filter(pl.col("map_name") == mp), yb[m], pb[m])
    # ---- snapshot model on the 2026 matches
    G26 = G.filter(pl.col("era") == "2026")
    sdir = os.path.join(W.ROOT, "data", "status_1hz_scopetest")
    G26 = W.drop_roster_defect_rounds(G26, "scopetest")
    G26 = FS.drop_inconsistent_rounds(G26, sdir)
    G26 = G26.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Int8).alias(f"map_is_{m}") for m in MAPS])
    G26 = G26.with_columns(FS.loadout_features(G26, sdir))
    bs, fs, i_s = booster(a.name)
    ps = predict(bs, fs, i_s, G26)
    ys = G26["label_ct_win"].to_numpy().astype(np.int64)
    out["snapshot_all"] = block(G26, ys, ps)
    out["snapshot_by_map"] = {}
    for mp in ("de_vertigo", "de_overpass"):
        m = (G26["map_name"] == mp).to_numpy()
        if m.sum() > 1000:
            out["snapshot_by_map"][mp] = block(G26.filter(pl.col("map_name") == mp), ys[m], ps[m])
    # ---- the same models on their own held-out matches (reference)
    e = json.load(open(os.path.join(W.ROOT, "data", "combined", f"export_{a.name}.json"), encoding="utf-8"))
    eb = json.load(open(os.path.join(W.ROOT, "data", "combined", f"export_{a.basic}.json"), encoding="utf-8"))
    out["reference_snapshot_known_maps"] = {k: e["gpu_trained"][k] for k in ("logloss", "logloss_ci95", "auc", "ece10")} | {"n_matches": e["test"]["matches"]}
    out["reference_fallback_known_maps"] = {k: eb["gpu_trained"][k] for k in ("logloss", "logloss_ci95", "auc", "ece10")} | {"n_matches": eb["test"]["matches"]}
    out["note"] = "map one-hots are all zero for these maps; 2024 matches use the alive-count model only (no per-tick player table in that era); the serving code flags every one of these as out_of_scope"
    W.dump_json(out, os.path.join(W.ROOT, "data", "combined", f"eval_scope_maps_{a.name}.json"))
    for k in ("snapshot_all", "fallback_all", "reference_snapshot_known_maps", "reference_fallback_known_maps"):
        v = out[k]
        print(f"{k}: log loss {v['logloss']:.4f} {[round(x, 4) for x in v['logloss_ci95']]}  AUC {v['auc']:.3f}  ECE {v['ece10']:.4f}  matches {v['n_matches']}")
    for k in ("snapshot_by_map", "fallback_by_map"):
        for mp, v in out[k].items():
            print(f"  {k} {mp}: log loss {v['logloss']:.4f} {[round(x, 4) for x in v['logloss_ci95']]} matches {v['n_matches']}")


if __name__ == "__main__":
    main()
