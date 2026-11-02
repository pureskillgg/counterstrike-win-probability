"""Hyperparameter search for the best measured feature set: 58 inputs (59 minus the heavy group, plus score lead).

Same rows, frozen split and monotone constraints as the 59-input models (score lead unconstrained). Each configuration is trained with
early stopping on validation; the selected setting is the lowest validation log loss. Test is reported for every configuration, paired with
the baseline (depth 4, min child weight 20, subsample 0.8, eta 0.05) on the same seconds, but selection uses validation only.

Selected model saved as models/winprob_v8_tuned.json with metadata, served: false.

  ...python.exe -I -X utf8 scripts\\model\\exp_hparams58.py
Writes data/combined/exp_hparams58.json.
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
HEAVY = ["st_heavy_ct", "st_heavy_t"]
CONFIGS = [
    ("baseline d4 mcw20 sub0.8", dict(depth=4, mcw=20, subsample=0.8)),
    ("depth 5", dict(depth=5, mcw=20, subsample=0.8)),
    ("depth 6", dict(depth=6, mcw=20, subsample=0.8)),
    ("depth 8", dict(depth=8, mcw=20, subsample=0.8)),
    ("min child weight 50", dict(depth=4, mcw=50, subsample=0.8)),
    ("min child weight 100", dict(depth=4, mcw=100, subsample=0.8)),
    ("subsample 1.0", dict(depth=4, mcw=20, subsample=1.0)),
    ("colsample 0.8", dict(depth=4, mcw=20, subsample=0.8, colsample_bytree=0.8)),
    ("depth 6, mcw 100", dict(depth=6, mcw=100, subsample=0.8)),
    ("depth 6, mcw 100, colsample 0.8", dict(depth=6, mcw=100, subsample=0.8, colsample_bytree=0.8)),
    ("eta 0.03", dict(depth=4, mcw=20, subsample=0.8, eta=0.03)),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    feats = [f for f in feats59 if f not in HEAVY] + ["score_lead_diag"]
    mono = [mono59.get(f, 0) for f in feats]

    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    lead = (pl.scan_parquet(os.path.join(d, "grid.parquet")).filter(pl.col("era") == "2026").select(["match_key", "round", "score_lead_diag"])
            .unique(subset=["match_key", "round"]).collect())
    G = G.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf
    _, Xtr, ytr = W.arrays(G, tr, feats, stride=1)
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
    print(f"58 inputs; train {len(ytr):,} seconds, validation {len(yva):,}, test {len(yte):,} ({time.time() - t0:.0f}s)", flush=True)

    out = {"features": len(feats), "configs": []}
    base_pte = base_pva = None
    best = None
    models = {}
    for name, cfg in CONFIGS:
        mm, best_it = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=0, device="cuda", **cfg)
        pva, pte = W.predict(mm, best_it, Xva), W.predict(mm, best_it, Xte)
        if base_pte is None:
            base_pte, base_pva = pte, pva
            ev = W.evaluate(sub_te, yte, pte)
            evv = W.evaluate(sub_va, yva, pva)
        else:
            ev = W.evaluate(sub_te, yte, pte, ref=base_pte)
            evv = W.evaluate(sub_va, yva, pva, ref=base_pva)
        r = {"name": name, "config": cfg, "best_iteration": int(best_it), "validation_logloss": evv["logloss"],
             "test_logloss": ev["logloss"], "validation_delta_vs_baseline": evv.get("delta_vs_ref"),
             "test_delta_vs_baseline": ev.get("delta_vs_ref"), "test_delta_ci95": ev.get("delta_vs_ref_ci95")}
        out["configs"].append(r)
        models[name] = (mm, best_it, cfg)
        if best is None or r["validation_logloss"] < best["validation_logloss"]:
            best = r
        dv = r["validation_delta_vs_baseline"]
        dt = r["test_delta_vs_baseline"]
        ci = r["test_delta_ci95"]
        print(f"{name}: best iteration {best_it}; validation {r['validation_logloss']:.6f}"
              + (f" (delta {dv:+.5f})" if dv is not None else "") + f"; test {r['test_logloss']:.6f}"
              + (f" (delta {dt:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}])" if dt is not None else "") + f" ({time.time() - t0:.0f}s)", flush=True)
        W.dump_json(out, os.path.join(d, "exp_hparams58.json"))

    out["selected"] = best["name"]
    W.dump_json(out, os.path.join(d, "exp_hparams58.json"))
    print(f"selected by validation: {best['name']}: validation {best['validation_logloss']:.6f}; test {best['test_logloss']:.6f}", flush=True)

    mm, best_it, cfg = models[best["name"]]
    mm.get_booster().save_model(os.path.join(W.ROOT, "models", "winprob_v8_tuned.json"))
    meta = {"version": "winprob_v8_tuned", "features": feats, "monotone_constraints": dict(zip(feats, mono)), "best_iteration": int(best_it),
            "hyperparameters": cfg, "selection": best["name"], "selection_rule": "lowest validation log loss over the configurations in exp_hparams58.py",
            "xgboost": xgb.__version__, "device": "cuda", "split": "random", "split_manifest_sha256": sha(os.path.join(d, "split_manifest.json")),
            "drop_roster_defect_rounds": True, "validation_logloss": best["validation_logloss"], "test_logloss": best["test_logloss"],
            "test": {"rows": int(len(yte))}, "served": False,
            "note": "58 inputs (59 minus heavy, plus score lead) with the selected hyperparameters; needs per-tick player status to serve"}
    with open(os.path.join(W.ROOT, "models", "winprob_v8_tuned.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"saved models/winprob_v8_tuned.json ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
