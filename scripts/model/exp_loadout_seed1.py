"""Seed check for the top loadout candidates: does the ranking hold on a second XGBoost seed?

Same frozen split (seed 0 split, manifest sha256 a17f219b...). Only the XGBoost seed changes (seed 1, as winprob_v4_s1). Candidates: the control
(all 59), drop heavy (57), and the validation-selected set (38 inputs: drop heavy, ak/m4, awp, no primary, grenades). Each seed-1 cell is
paired with the seed-1 control on the same seconds.

  ...python.exe -I -X utf8 scripts\\model\\exp_loadout_seed1.py
Writes data/combined/exp_loadout_seed1.json.
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
AWP = ["st_awp_ct", "st_awp_t", "st_awp_diff"]
NO_PRIMARY = ["st_no_primary_ct", "st_no_primary_t"]
CANDIDATES = {
    "control (all 59)": [],
    "drop heavy (2)": HEAVY,
    "drop heavy + ak/m4 + awp + no primary + grenades (21)": HEAVY + AK_M4 + AWP + NO_PRIMARY + GRENADES,
}
SEED = 1


def main():
    t0 = time.time()
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}

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
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=SEED, device="cuda")
        del Xtr, ytr
        pte, pva = W.predict(m, best, Xte), W.predict(m, best, Xva)
        if ctrl is None:
            ev, evv = W.evaluate(sub_te, yte, pte), W.evaluate(sub_va, yva, pva)
        else:
            ev, evv = W.evaluate(sub_te, yte, pte, ref=ctrl["pte"]), W.evaluate(sub_va, yva, pva, ref=ctrl["pva"])
        res = {"name": name, "n_features": len(feats), "seed": SEED, "best_iteration": int(best),
               "validation_logloss": evv["logloss"], "test_logloss": ev["logloss"]}
        if ctrl is not None:
            res.update({"validation_delta_vs_control": evv["delta_vs_ref"], "test_delta_vs_control": ev["delta_vs_ref"],
                        "test_delta_ci95": ev["delta_vs_ref_ci95"]})
        return res, {"pte": pte, "pva": pva}

    out = {"seed": SEED, "variants": []}
    control, cp = fit_eval("control (all 59)", feats59)
    out["variants"].append(control)
    print(f"control seed {SEED}: validation {control['validation_logloss']:.6f}; test {control['test_logloss']:.6f} ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_loadout_seed1.json"))
    for name, cols in CANDIDATES.items():
        if not cols:
            continue
        feats = [f for f in feats59 if f not in cols]
        r, _ = fit_eval(name, feats, cp)
        out["variants"].append(r)
        print(f"{name} ({r['n_features']} inputs), seed {SEED}: validation {r['validation_logloss']:.6f} "
              f"(delta {r['validation_delta_vs_control']:+.5f}); test {r['test_logloss']:.6f} "
              f"(delta {r['test_delta_vs_control']:+.5f} [{r['test_delta_ci95'][0]:+.5f}, {r['test_delta_ci95'][1]:+.5f}]) ({time.time() - t0:.0f}s)",
              flush=True)
        W.dump_json(out, os.path.join(d, "exp_loadout_seed1.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
