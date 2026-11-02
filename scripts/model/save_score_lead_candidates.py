"""Retrain and save the two score-lead candidates. Neither changes the served model.

(a) winprob_v7_59_drop_heavy_scorelead: the 59 inputs minus the heavy group, plus score lead (58 inputs). The best measured 59-input variant
    (exp_scorelead_combo.py). Same rows and split as winprob_v4 (cache_snapshot_2026, frozen split). Needs per-tick player status to serve.
(b) winprob_v7_basic_scorelead: the served 13 inputs plus score lead (14 inputs), with the served model's rows and settings (all matches,
    grid, roster-defect rounds dropped, monotone alive counts, score lead unconstrained). Servable once the serving function reads score lead.

Both: depth 4, eta 0.05, min child weight 20, subsample 0.8, early stopping after 100 rounds on validation, seed 0, GPU. Score lead is the
grid's score_lead_diag (freeze-end CT score minus T score), first row per match and round, nulls as 0 (counted in the metadata).

  ...python.exe -I -X utf8 scripts\\model\\save_score_lead_candidates.py
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
import xgboost as xgb

import train_final as T
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
HEAVY = ["st_heavy_ct", "st_heavy_t"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def lead_frame(d, eras=None):
    lf = pl.scan_parquet(os.path.join(d, "grid.parquet"))
    if eras is not None:
        lf = lf.filter(pl.col("era").is_in(eras))
    return lf.select(["match_key", "round", "score_lead_diag"]).unique(subset=["match_key", "round"]).collect()


def save(name, m, feats, mono, best, ll_va, ll_te, n_te, extra, d):
    m.get_booster().save_model(os.path.join(W.ROOT, "models", f"{name}.json"))
    meta = {"version": name, "features": feats, "monotone_constraints": dict(zip(feats, mono)), "best_iteration": int(best),
            "params": {"depth": 4, "eta": 0.05, "mcw": 20, "subsample": 0.8, "early_stopping_rounds": 100, "seed": 0},
            "xgboost": xgb.__version__, "device": "cuda", "split": "random",
            "split_manifest_sha256": sha(os.path.join(d, "split_manifest.json")), "drop_roster_defect_rounds": True,
            "validation_logloss": ll_va, "test_logloss": ll_te, "test": {"rows": n_te}, "served": False, **extra}
    with open(os.path.join(W.ROOT, "models", f"{name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def main():
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}

    # (a) 58 inputs on the 2026 cache
    f58 = [f for f in feats59 if f not in HEAVY] + ["score_lead_diag"]
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    G = G.join(lead_frame(d, ["2026"]), on=["match_key", "round"], how="left")
    miss_a = int(G["score_lead_diag"].is_null().sum())
    G = G.with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf
    mono_a = [mono59.get(f, 0) for f in f58]
    _, Xtr, ytr = W.arrays(G, tr, f58, stride=1)
    sub_va, Xva, yva = W.arrays(G, va, f58, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, f58, stride=1)
    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono_a, seed=0, device="cuda")
    ll_va = W.evaluate(sub_va, yva, W.predict(m, best, Xva))["logloss"]
    ll_te = W.evaluate(sub_te, yte, W.predict(m, best, Xte))["logloss"]
    print(f"(a) 58 inputs: validation {ll_va:.6f}; test {ll_te:.6f} (test rows {len(yte):,}); score lead nulls {miss_a} ({time.time() - t0:.0f}s)", flush=True)
    save("winprob_v7_59_drop_heavy_scorelead", m, f58, mono_a, best, ll_va, ll_te, len(yte),
         {"selection": "best measured 59-input variant (score lead, heavy dropped); needs per-tick player status to serve",
          "rows_without_grid_score_lead": miss_a, "training": {"matches": len(tr), "rows": int(len(ytr))}}, d)
    del G, Xtr, ytr, Xva, Xte, sub_va, sub_te, m

    # (b) the served 13 inputs plus score lead, on the served model's rows (all matches)
    Gb, feats_b = T.load_basic("combined")
    Gb = Gb.join(lead_frame(d), on=["match_key", "round"], how="left")
    miss_b = int(Gb["score_lead_diag"].is_null().sum())
    Gb = Gb.with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    fb = feats_b + ["score_lead_diag"]
    served_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4_basic.meta.json"), encoding="utf-8"))
    mono_b = [int(served_meta["monotone_constraints"][f]) if f in served_meta["monotone_constraints"] else 0 for f in fb]
    spb = W.make_split("random", Gb, seed=0)
    vab, teb, trb = sorted(spb["val"]), sorted(spb["test"]), sorted(spb["train"])
    _, Xtrb, ytrb = W.arrays(Gb, trb, fb, stride=1)
    sub_vab, Xvab, yvab = W.arrays(Gb, vab, fb, stride=1)
    sub_teb, Xteb, yteb = W.arrays(Gb, teb, fb, stride=1)
    mb, bestb = W.fit_xgb(Xtrb, ytrb, Xvab, yvab, mono_b, seed=0, device="cuda")
    ll_vab = W.evaluate(sub_vab, yvab, W.predict(mb, bestb, Xvab))["logloss"]
    ll_teb = W.evaluate(sub_teb, yteb, W.predict(mb, bestb, Xteb))["logloss"]
    print(f"(b) served 13 + score lead (14 inputs): validation {ll_vab:.6f} (served {served_meta['validation_logloss']:.6f}); "
          f"test {ll_teb:.6f} (served {served_meta['test_logloss']:.6f}); test rows {len(yteb):,} (served {served_meta['test']['rows']:,}); "
          f"score lead nulls {miss_b} ({time.time() - t0:.0f}s)", flush=True)
    save("winprob_v7_basic_scorelead", mb, fb, mono_b, bestb, ll_vab, ll_teb, len(yteb),
         {"selection": "served 13 inputs plus score lead; not served until the serving function reads score lead and the parity test passes",
          "rows_without_grid_score_lead": miss_b, "served_reference": "winprob_v4_basic",
          "served_reference_test_logloss": served_meta["test_logloss"], "served_reference_validation_logloss": served_meta["validation_logloss"]}, d)
    print(f"saved both candidates ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
