"""Joint drop: the 59-input model with the 32 loadout features judged redundant removed together (27 inputs remain).

Kept loadout features (14): team health (st_hp_ct, st_hp_t, st_hp_diff), defuse kits, defusing and helmets (st_kit_n, st_defusing,
st_helmet_n, both sides), armour (st_armor_n, st_armor_sum, both sides, st_armor_n_diff). Dropped (32): all rifle features (incl. AK/M4),
AWP, SMG, other snipers, no-primary, shotguns and machine guns (heavy), grenades (10), equipment value (3).

Same settings and frozen split as winprob_v4. Control: a retrain with all 59 inputs, scored on the same seconds. Decision rule, set before
the run: adopt the 27-input model only if its validation log loss is within 0.0002 of the control. Test is reported after that decision.

  ...python.exe -I -X utf8 scripts\\model\\exp_loadout_joint.py
Writes data/combined/exp_loadout_joint.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
DROP = [
    "st_heavy_ct", "st_heavy_t",
    "st_rifle_ct", "st_rifle_t", "st_rifle_diff", "st_ak_ct", "st_ak_t", "st_m4_ct", "st_m4_t",
    "st_awp_ct", "st_awp_t", "st_awp_diff",
    "st_smg_ct", "st_smg_t", "st_smg_diff",
    "st_sniper_other_ct", "st_sniper_other_t",
    "st_no_primary_ct", "st_no_primary_t",
    *[f"st_{g}_{s}" for g in ["flash", "smoke", "he", "fire", "decoy"] for s in ["ct", "t"]],
    "st_equip_ct", "st_equip_t", "st_equip_diff",
]
ADOPT_MARGIN = 0.0002


def main():
    t0 = time.time()
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    missing = [c for c in DROP if c not in feats59]
    if missing or len(DROP) != 32 or len(set(DROP)) != 32:
        sys.exit(f"drop list problem: missing {missing}, length {len(DROP)}")
    keep = [f for f in feats59 if f not in DROP]
    print(f"dropping {len(DROP)} features, keeping {len(keep)} ({len(keep) - 13} loadout)", flush=True)

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
        res = {"name": name, "n_features": len(feats), "best_iteration": int(best),
               "test_logloss": ev["logloss"], "validation_logloss": evv["logloss"]}
        if ctrl is not None:
            res.update({"test_delta_vs_control": ev["delta_vs_ref"], "test_delta_ci95": ev["delta_vs_ref_ci95"],
                        "validation_delta_vs_control": evv["delta_vs_ref"], "validation_delta_ci95": evv["delta_vs_ref_ci95"]})
        return res, {"pte": pte, "pva": pva}

    out = {"decision_rule": f"adopt if validation delta < {ADOPT_MARGIN}", "drop": DROP, "keep": keep, "variants": []}
    control, cp = fit_eval("control (all 59 inputs)", feats59)
    out["variants"].append(control)
    print(f"control: validation {control['validation_logloss']:.6f}; test {control['test_logloss']:.6f} ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_loadout_joint.json"))

    joint, _ = fit_eval("joint drop (27 inputs)", keep, cp)
    out["variants"].append(joint)
    adopt = joint["validation_delta_vs_control"] < ADOPT_MARGIN
    out["decision"] = "adopt" if adopt else "keep the 59-input model"
    lo, hi = joint["test_delta_ci95"]
    print(f"joint drop (27 inputs): validation delta {joint['validation_delta_vs_control']:+.5f}; "
          f"test delta {joint['test_delta_vs_control']:+.5f} [{lo:+.5f}, {hi:+.5f}]; decision: {out['decision']} ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_loadout_joint.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
