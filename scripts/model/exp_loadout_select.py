"""Select the loadout set with the best validation score, then save it.

Candidates (every one is the 59-input model minus the listed features, same settings and frozen split as winprob_v4):
  control: all 59 inputs (retrained, the paired reference)
  drop ak + m4 split (4)
  drop heavy (2)
  drop all rifles (7)
  drop equipment + grenades (13)
  drop heavy + ak/m4 + awp + no primary + grenades (21): the single drops whose validation delta was <= 0, combined
  same, with all rifles instead of ak/m4 (24)

Selection rule: lowest validation log loss, decided before the test column is read. Test is reported for every candidate for transparency.
The chosen set is saved as models/winprob_v6_selected.json with its metadata, served: false.

  ...python.exe -I -X utf8 scripts\\model\\exp_loadout_select.py
Writes data/combined/exp_loadout_select.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
EQUIP = ["st_equip_ct", "st_equip_t", "st_equip_diff"]
GRENADES = [f"st_{g}_{s}" for g in ["flash", "smoke", "he", "fire", "decoy"] for s in ["ct", "t"]]
HEAVY = ["st_heavy_ct", "st_heavy_t"]
AK_M4 = ["st_ak_ct", "st_ak_t", "st_m4_ct", "st_m4_t"]
ALL_RIFLES = ["st_rifle_ct", "st_rifle_t", "st_rifle_diff"] + AK_M4
AWP = ["st_awp_ct", "st_awp_t", "st_awp_diff"]
NO_PRIMARY = ["st_no_primary_ct", "st_no_primary_t"]
CANDIDATES = {
    "control (all 59)": [],
    "drop ak + m4 split (4)": AK_M4,
    "drop heavy (2)": HEAVY,
    "drop all rifles (7)": ALL_RIFLES,
    "drop equipment + grenades (13)": EQUIP + GRENADES,
    "drop heavy + ak/m4 + awp + no primary + grenades (21)": HEAVY + AK_M4 + AWP + NO_PRIMARY + GRENADES,
    "drop heavy + all rifles + awp + no primary + grenades (24)": HEAVY + ALL_RIFLES + AWP + NO_PRIMARY + GRENADES,
}


def main():
    t0 = time.time()
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    for name, cols in CANDIDATES.items():
        missing = [c for c in cols if c not in feats59]
        if missing:
            sys.exit(f"{name}: not among the 59 inputs: {missing}")

    d = os.path.join(W.ROOT, "data", "combined")
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf
    print(f"2026 seconds {G.height:,}; train {len(tr)} matches, validation {len(va)}, test {len(te)} ({time.time() - t0:.0f}s)", flush=True)

    def fit_eval(name, feats, ctrl=None):
        mono = [mono59.get(f, 0) for f in feats]
        _, Xtr, ytr = W.arrays(G, tr, feats, stride=1)
        sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
        sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=0, device="cuda")
        del Xtr, ytr
        pte, pva = W.predict(m, best, Xte), W.predict(m, best, Xva)
        if ctrl is None:
            ev, evv = W.evaluate(sub_te, yte, pte), W.evaluate(sub_va, yva, pva)
        else:
            ev, evv = W.evaluate(sub_te, yte, pte, ref=ctrl["pte"]), W.evaluate(sub_va, yva, pva, ref=ctrl["pva"])
        res = {"name": name, "features": feats, "n_features": len(feats), "best_iteration": int(best),
               "validation_logloss": evv["logloss"], "test_logloss": ev["logloss"], "test_rows": len(yte)}
        if ctrl is not None:
            res.update({"validation_delta_vs_control": evv["delta_vs_ref"], "test_delta_vs_control": ev["delta_vs_ref"],
                        "test_delta_ci95": ev["delta_vs_ref_ci95"]})
        return res, m, {"pte": pte, "pva": pva}

    out = {"selection_rule": "lowest validation log loss; test reported for every candidate", "candidates": []}
    boosters = {}
    control_res, control_m, cp = fit_eval("control (all 59)", feats59)
    out["candidates"].append(control_res)
    boosters[control_res["name"]] = (control_m, feats59)
    print(f"control: validation {control_res['validation_logloss']:.6f}; test {control_res['test_logloss']:.6f} ({time.time() - t0:.0f}s)", flush=True)
    for name, cols in CANDIDATES.items():
        if not cols:
            continue
        feats = [f for f in feats59 if f not in cols]
        r, m, _ = fit_eval(name, feats, cp)
        out["candidates"].append(r)
        boosters[name] = (m, feats)
        print(f"{name} ({r['n_features']} inputs): validation {r['validation_logloss']:.6f} "
              f"(delta {r['validation_delta_vs_control']:+.5f}); test {r['test_logloss']:.6f} "
              f"(delta {r['test_delta_vs_control']:+.5f} [{r['test_delta_ci95'][0]:+.5f}, {r['test_delta_ci95'][1]:+.5f}]) ({time.time() - t0:.0f}s)",
              flush=True)
        W.dump_json(out, os.path.join(d, "exp_loadout_select.json"))

    best = min(out["candidates"], key=lambda r: r["validation_logloss"])
    out["selected"] = best["name"]
    print(f"selected by validation: {best['name']} ({best['n_features']} inputs): validation {best['validation_logloss']:.6f}; "
          f"test {best['test_logloss']:.6f}", flush=True)
    W.dump_json(out, os.path.join(d, "exp_loadout_select.json"))

    m, feats = boosters[best["name"]]
    mpath = os.path.join(W.ROOT, "models", "winprob_v6_selected.json")
    m.get_booster().save_model(mpath)
    meta = {"version": "winprob_v6_selected", "selection": best["name"], "features": feats,
            "monotone_constraints": {f: mono59.get(f, 0) for f in feats}, "best_iteration": best["best_iteration"],
            "params": {"depth": 4, "eta": 0.05, "mcw": 20, "subsample": 0.8, "early_stopping_rounds": 100, "seed": 0},
            "split": "random", "split_manifest_sha256": ref_meta["split_manifest_sha256"], "drop_roster_defect_rounds": True,
            "validation_logloss": best["validation_logloss"], "test_logloss": best["test_logloss"], "test_rows": best["test_rows"],
            "selection_rule": out["selection_rule"], "served": False,
            "note": "loadout selection by validation; needs per-tick player status to serve, so not the served default"}
    with open(os.path.join(W.ROOT, "models", "winprob_v6_selected.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"saved models/winprob_v6_selected.json ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
