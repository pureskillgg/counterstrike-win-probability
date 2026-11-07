"""Save the consistent loadout choice: the 59-input model without the heavy group (shotguns and machine guns), 57 inputs.

Chosen because it is the only candidate that beats the control on both splits for both XGBoost seeds (exp_loadout_select.json, exp_loadout_seed1.json).
Same settings and frozen split as winprob_v4. The retrain should reproduce the seed-0 numbers recorded there (validation 0.453557, test 0.456411).
Saved as models/winprob_v6_drop_heavy.json with metadata, served: true (the 20 Oct 2026 decision, pin aaecad58).

  ...python.exe -I -X utf8 scripts\\model\\save_winprob_v6_drop_heavy.py
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


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    t0 = time.time()
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    feats = [f for f in feats59 if f not in HEAVY]
    mono = [mono59.get(f, 0) for f in feats]

    d = os.path.join(W.ROOT, "data", "combined")
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf
    _, Xtr, ytr = W.arrays(G, tr, feats, stride=1)
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=0, device="cuda")
    pva, pte = W.predict(m, best, Xva), W.predict(m, best, Xte)
    ll_va = W.evaluate(sub_va, yva, pva)["logloss"]
    ll_te = W.evaluate(sub_te, yte, pte)["logloss"]
    print(f"drop heavy (57 inputs), seed 0: best iteration {best}; validation {ll_va:.6f} (recorded 0.453557); "
          f"test {ll_te:.6f} (recorded 0.456411); test rows {len(yte):,} ({time.time() - t0:.0f}s)", flush=True)

    mpath = os.path.join(W.ROOT, "models", "winprob_v6_drop_heavy.json")
    m.get_booster().save_model(mpath)
    meta = {"version": "winprob_v6_drop_heavy", "selection": "drop heavy (shotguns, machine guns): the only candidate better than the control on both splits for both seeds",
            "features": feats, "monotone_constraints": {f: mono59.get(f, 0) for f in feats}, "best_iteration": int(best),
            "params": {"depth": 4, "eta": 0.05, "mcw": 20, "subsample": 0.8, "early_stopping_rounds": 100, "seed": 0},
            "xgboost": xgb.__version__, "device": "cuda", "split": "random", "split_manifest_sha256": sha(os.path.join(d, "split_manifest.json")),
            "drop_roster_defect_rounds": True, "input_file": "cache_snapshot_2026.parquet",
            "training": {"matches": len(tr), "val_matches": len(va), "rows": int(len(ytr))},
            "test": {"matches": len(te), "rows": int(len(yte))},
            "validation_logloss": ll_va, "test_logloss": ll_te, "served": True,
            "note": "served default since the 20 Oct 2026 decision (pin aaecad58): 57 inputs, one XGBoost model, needs per-tick player status in the tome"}
    with open(os.path.join(W.ROOT, "models", "winprob_v6_drop_heavy.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"saved models/winprob_v6_drop_heavy.json ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
