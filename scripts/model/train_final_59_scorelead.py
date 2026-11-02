"""Final model for the winning configuration: the 59 snapshot inputs plus score lead (freeze-end CT score minus T score), trained under the same
rules as the other final models: the frozen split (data/combined/split_manifest.json), the baseline settings (depth 4, eta 0.05, min child
weight 20, subsample 0.8, seed 0), early stopping after 100 rounds on the validation seconds, the monotone constraints on the alive counts
(score lead unconstrained), every second, GPU.

Saves models/winprob_v5_59_scorelead.json and .meta.json, like train_final.py. It does NOT change the served model: the served function
reads only the event channels, and the 59-input features need per-tick player status, so this model is not servable yet.

  ...python.exe -I -X utf8 scripts\\model\\train_final_59_scorelead.py [--device cuda]
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--store", default="combined")
    a = ap.parse_args()
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", a.store)
    manifest = os.path.join(d, "split_manifest.json")
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    mono59 = [int(ref_meta["monotone_constraints"][f]) for f in feats59]
    name = "winprob_v5_59_scorelead"
    cache = os.path.join(d, "cache_snapshot_2026.parquet")
    grid = os.path.join(d, "grid.parquet")

    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    lf = pl.scan_parquet(cache).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    del kdf
    G = lf.with_columns(pl.col(feats59).cast(pl.Float32)).collect(engine="streaming")
    lead = (pl.scan_parquet(grid).filter(pl.col("era") == "2026").select(["match_key", "round", "score_lead_diag"])
            .unique(subset=["match_key", "round"]).collect())
    G = G.join(lead, on=["match_key", "round"], how="left")
    missing_lead = int(G["score_lead_diag"].is_null().sum())
    G = G.with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    feats = feats59 + ["score_lead_diag"]
    mono = mono59 + [0]
    _, Xtr, ytr = W.arrays(G, tr, feats, stride=1)
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
    del G
    print(f"{name}: train {len(tr)} matches / {len(ytr):,} seconds; validation {len(va)} / {len(yva):,}; test {len(te)} / {len(yte):,} "
          f"({time.time() - t0:.0f}s)", flush=True)

    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=0, device=a.device)
    print(f"trained: best iteration {best} ({time.time() - t0:.0f}s)", flush=True)
    pva = W.predict(m, best, Xva)
    pte = W.predict(m, best, Xte)
    ev_va = W.evaluate(sub_va, yva, pva)
    ev_te = W.evaluate(sub_te, yte, pte)
    print(f"validation log loss {ev_va['logloss']:.5f}; test log loss {ev_te['logloss']:.5f} "
          f"{[round(x, 5) for x in ev_te['logloss_ci95']]} AUC {ev_te['auc']:.4f} ECE10 {ev_te['ece10']:.4f}", flush=True)

    os.makedirs(os.path.join(W.ROOT, "models"), exist_ok=True)
    m.get_booster().save_model(os.path.join(W.ROOT, "models", f"{name}.json"))
    meta = {
        "version": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "features": feats,
        "monotone_constraints": dict(zip(feats, mono)), "best_iteration": int(best),
        "params": {"depth": 4, "eta": 0.05, "mcw": 20, "subsample": 0.8, "early_stopping_rounds": 100, "seed": 0},
        "xgboost": xgb.__version__, "device": a.device, "stride": 1, "split": "random", "split_manifest_sha256": sha256(manifest),
        "input_file": os.path.basename(cache), "input_file_sha256": sha256(cache), "score_lead_source": os.path.basename(grid),
        "score_lead_source_sha256": sha256(grid), "rows_without_grid_score_lead": missing_lead, "drop_roster_defect_rounds": True,
        "training": {"matches": len(tr), "val_matches": len(va), "rows": int(len(ytr))},
        "test": {"matches": len(te), "rows": int(len(yte))},
        "validation_logloss": ev_va["logloss"], "test_logloss": ev_te["logloss"], "test_logloss_ci95": ev_te["logloss_ci95"],
        "test_logloss_round_weighted": ev_te["logloss_round_weighted"], "test_auc": ev_te["auc"], "test_ece10": ev_te["ece10"],
        "test_brier": ev_te["brier"], "served": False,
        "note": "59 snapshot inputs plus score lead; needs per-tick player status to serve, so not the served default",
    }
    with open(os.path.join(W.ROOT, "models", f"{name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"done ({time.time() - t0:.0f}s): models/{name}.json", flush=True)


if __name__ == "__main__":
    main()
