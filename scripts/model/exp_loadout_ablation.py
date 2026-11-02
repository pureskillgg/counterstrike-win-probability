"""Loadout ablation for the 59-input model: retrain with one feature group removed, and once with a pistol count added.

Same settings and frozen split as winprob_v4 (depth 4, eta 0.05, min child weight 20, subsample 0.8, early stopping after 100 rounds on the
validation seconds, monotone alive counts, every second, GPU). A control retrain with all 59 inputs is the paired reference: every variant is
scored on the same test seconds, and the delta is the change in log loss against the control.

Sign convention (W.evaluate: delta = model loss minus reference loss):
  drop <group>:  positive delta = removing the group makes the model worse, so the group helps.
  add pistol:    negative delta = adding the pistol count helps.

Pistol count: alive players whose secondary weapon (inv_secondary, data/status_1hz) is a pistol code, per side (st_pistol_n_ct, _t, _diff).
Pistol codes: 1 deagle, 2 elite, 4 glock, 30 tec9, 32 p2000, 36 p250, 61 usp, 63 cz75, 64 revolver (0 = none, not counted).
Shotguns are in the `heavy` group with machine guns (both live in the same feature), so that group cannot separate them.

  ...python.exe -I -X utf8 scripts\\model\\exp_loadout_ablation.py
Writes data/combined/exp_loadout_ablation.json.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
CT, T = 3, 2
PISTOL = [1, 2, 4, 30, 32, 36, 61, 63, 64]
PISTOL_FEATS = ["st_pistol_n_ct", "st_pistol_n_t", "st_pistol_n_diff"]
GROUPS = {
    "heavy (shotguns, machine guns)": ["st_heavy_ct", "st_heavy_t"],
    "armor": ["st_armor_n_ct", "st_armor_sum_ct", "st_armor_n_t", "st_armor_sum_t", "st_armor_n_diff"],
    "ak + m4 split (keeps all rifles)": ["st_ak_ct", "st_ak_t", "st_m4_ct", "st_m4_t"],
    "all rifles (incl. ak, m4)": ["st_rifle_ct", "st_rifle_t", "st_rifle_diff", "st_ak_ct", "st_ak_t", "st_m4_ct", "st_m4_t"],
    "awp": ["st_awp_ct", "st_awp_t", "st_awp_diff"],
    "smg": ["st_smg_ct", "st_smg_t", "st_smg_diff"],
    "other snipers (scout, autos)": ["st_sniper_other_ct", "st_sniper_other_t"],
    "no primary (pistol-only players)": ["st_no_primary_ct", "st_no_primary_t"],
    "health": ["st_hp_ct", "st_hp_t", "st_hp_diff"],
    "equipment value": ["st_equip_ct", "st_equip_t", "st_equip_diff"],
    "grenades (flash, smoke, he, fire, decoy)": [f"st_{g}_{s}" for g in ["flash", "smoke", "he", "fire", "decoy"] for s in ["ct", "t"]],
    "defuse kits, defusing, helmets": [f"st_{g}_{s}" for g in ["kit_n", "defusing", "helmet_n"] for s in ["ct", "t"]],
}


def pistol_counts(mks):
    """Per (match, round, tick, side): alive players holding a pistol as the secondary weapon. Read one day file at a time."""
    parts = []
    for f in sorted(glob.glob(os.path.join(W.ROOT, "data", "status_1hz", "*.parquet"))):
        S = pl.read_parquet(f, columns=["match_key", "round", "tick", "side", "inv_secondary"]).filter(pl.col("match_key").is_in(mks))
        if S.height:
            parts.append(S.group_by(["match_key", "round", "tick", "side"]).agg(pl.col("inv_secondary").is_in(PISTOL).sum().alias("n")))
        del S
    A = pl.concat(parts)
    key = ["match_key", "round", "tick"]
    ct = A.filter(pl.col("side") == CT).drop("side").rename({"n": "st_pistol_n_ct"})
    t = A.filter(pl.col("side") == T).drop("side").rename({"n": "st_pistol_n_t"})
    return ct, t, key


def main():
    t0 = time.time()
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    for g, cols in GROUPS.items():
        missing = [c for c in cols if c not in feats59]
        if missing:
            sys.exit(f"{g}: not among the 59 inputs: {missing}")

    d = os.path.join(W.ROOT, "data", "combined")
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf

    ct, t, key = pistol_counts(set(G["match_key"].unique().to_list()))
    G = G.join(ct, on=key, how="left").join(t, on=key, how="left")
    G = G.with_columns(pl.col("st_pistol_n_ct", "st_pistol_n_t").fill_null(0).cast(pl.Float32))
    G = G.with_columns((pl.col("st_pistol_n_ct") - pl.col("st_pistol_n_t")).alias("st_pistol_n_diff"))
    print(f"2026 seconds {G.height:,}; train {len(tr)} matches, validation {len(va)}, test {len(te)} ({time.time() - t0:.0f}s)", flush=True)

    def fit_eval(name, feats, ctrl=None):
        mono = [mono59.get(f, 0) for f in feats]
        sub_tr, Xtr, ytr = W.arrays(G, tr, feats, stride=1)
        sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
        sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=0, device="cuda")
        del Xtr, ytr, sub_tr
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

    out = {"settings": "winprob_v4 settings, frozen split, every second; deltas vs the retrained control (positive = dropping hurts)",
           "reference_winprob_v4": {"test_logloss": ref_meta["test_logloss"], "validation_logloss": ref_meta["validation_logloss"]},
           "variants": []}
    control, cp = fit_eval("control (all 59 inputs)", feats59)
    out["variants"].append(control)
    print(f"control: validation {control['validation_logloss']:.6f} (winprob_v4 {ref_meta['validation_logloss']:.6f}); "
          f"test {control['test_logloss']:.6f} (winprob_v4 {ref_meta['test_logloss']:.6f}) ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_loadout_ablation.json"))

    for g, cols in GROUPS.items():
        feats = [f for f in feats59 if f not in cols]
        r, _ = fit_eval(f"drop {g}", feats, cp)
        out["variants"].append(r)
        lo, hi = r["test_delta_ci95"]
        print(f"drop {g}: test delta {r['test_delta_vs_control']:+.5f} [{lo:+.5f}, {hi:+.5f}] "
              f"validation delta {r['validation_delta_vs_control']:+.5f} ({time.time() - t0:.0f}s)", flush=True)
        W.dump_json(out, os.path.join(d, "exp_loadout_ablation.json"))

    r, _ = fit_eval("add pistol count (3 features)", feats59 + PISTOL_FEATS, cp)
    out["variants"].append(r)
    lo, hi = r["test_delta_ci95"]
    print(f"add pistol count: test delta {r['test_delta_vs_control']:+.5f} [{lo:+.5f}, {hi:+.5f}] "
          f"validation delta {r['validation_delta_vs_control']:+.5f} ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_loadout_ablation.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
