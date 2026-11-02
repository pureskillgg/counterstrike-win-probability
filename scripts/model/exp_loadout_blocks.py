"""Block-level joint drops: which block of the 32 redundant loadout features carries the validation loss found by exp_loadout_joint.py.

Blocks (the 32 dropped in exp_loadout_joint.py are the union of the first two):
  equipment + grenades (13): equipment value (3) and grenades (10)
  weapon types (19):         all rifles incl. AK/M4 (7), AWP (3), SMG (3), other snipers (2), no primary (2), heavy (2)
  equipment only (3):        equipment value, to separate it from the grenades

Each variant removes one block from the 59 inputs and is paired with a retrained control on the same seconds. Same settings and frozen
split as winprob_v4. Positive delta = removing the block hurts.

  ...python.exe -I -X utf8 scripts\\model\\exp_loadout_blocks.py [--set followup]
Writes data/combined/exp_loadout_blocks.json (or exp_loadout_blocks_followup.json with --set followup).
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
WEAPONS = ["st_heavy_ct", "st_heavy_t",
           "st_rifle_ct", "st_rifle_t", "st_rifle_diff", "st_ak_ct", "st_ak_t", "st_m4_ct", "st_m4_t",
           "st_awp_ct", "st_awp_t", "st_awp_diff",
           "st_smg_ct", "st_smg_t", "st_smg_diff",
           "st_sniper_other_ct", "st_sniper_other_t",
           "st_no_primary_ct", "st_no_primary_t"]
BLOCKS = {
    "drop equipment + grenades (13)": EQUIP + GRENADES,
    "drop weapon types (19)": WEAPONS,
    "drop equipment only (3)": EQUIP,
}
# follow-up: with the weapon types gone, which of equipment or grenades carries the joint loss
FOLLOWUP = {
    "drop weapon types + equipment (22)": WEAPONS + EQUIP,
    "drop weapon types + grenades (29)": WEAPONS + GRENADES,
}


def main(which="first"):
    t0 = time.time()
    blocks = BLOCKS if which == "first" else FOLLOWUP
    outname = "exp_loadout_blocks.json" if which == "first" else "exp_loadout_blocks_followup.json"
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    for name, cols in blocks.items():
        missing = [c for c in cols if c not in feats59]
        if missing:
            sys.exit(f"{name}: not among the 59 inputs: {missing}")
    if which == "first" and len(BLOCKS["drop equipment + grenades (13)"]) + len(BLOCKS["drop weapon types (19)"]) != 32:
        sys.exit("the two blocks must make up the 32 features of exp_loadout_joint.py")

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

    out = {"settings": "winprob_v4 settings, frozen split, every second; deltas vs the retrained control (positive = removing the block hurts)",
           "variants": []}
    control, cp = fit_eval("control (all 59 inputs)", feats59)
    out["variants"].append(control)
    print(f"control: validation {control['validation_logloss']:.6f}; test {control['test_logloss']:.6f} ({time.time() - t0:.0f}s)", flush=True)
    W.dump_json(out, os.path.join(d, "exp_loadout_blocks.json"))

    for name, cols in blocks.items():
        feats = [f for f in feats59 if f not in cols]
        r, _ = fit_eval(name, feats, cp)
        out["variants"].append(r)
        lo, hi = r["test_delta_ci95"]
        print(f"{name}: test delta {r['test_delta_vs_control']:+.5f} [{lo:+.5f}, {hi:+.5f}]; "
              f"validation delta {r['validation_delta_vs_control']:+.5f} ({time.time() - t0:.0f}s)", flush=True)
        W.dump_json(out, os.path.join(d, "exp_loadout_blocks.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="first", choices=["first", "followup"], help="first: the three blocks; followup: weapon types with equipment or grenades")
    main(ap.parse_args().set)
