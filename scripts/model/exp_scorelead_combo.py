"""Score lead against the chosen loadout set: is the best combination better than winprob_v4 and winprob_v6_drop_heavy?

Variants per XGBoost seed (0 and 1), all on the frozen split, same settings as winprob_v4:
  control: the 59 inputs (paired reference)
  + score lead: the 59 inputs plus score_lead_diag (freeze-end CT score minus T score), unconstrained
  + score lead, drop heavy: the 57 inputs without the heavy group, plus score_lead_diag (58 inputs)
Each variant is paired with the control of the same seed on the same seconds (delta = model minus control; negative = better).
Score lead is joined from data/combined/grid.parquet as in train_final_59_scorelead.py (first row per match and round, 2026, nulls as 0).

  ...python.exe -I -X utf8 scripts\\model\\exp_scorelead_combo.py
Writes data/combined/exp_scorelead_combo.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
HEAVY = ["st_heavy_ct", "st_heavy_t"]
SEEDS = [0, 1]


def main():
    t0 = time.time()
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}

    d = os.path.join(W.ROOT, "data", "combined")
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
    print(f"2026 seconds {G.height:,}; train {len(tr)} matches, validation {len(va)}, test {len(te)} ({time.time() - t0:.0f}s)", flush=True)

    feats_sl = feats59 + ["score_lead_diag"]
    feats_sl_dh = [f for f in feats59 if f not in HEAVY] + ["score_lead_diag"]
    variants = [("control (59)", feats59), ("+ score lead (60)", feats_sl), ("+ score lead, drop heavy (58)", feats_sl_dh)]

    def fit_eval(name, feats, seed, ctrl=None):
        mono = [mono59.get(f, 0) for f in feats]
        _, Xtr, ytr = W.arrays(G, tr, feats, stride=1)
        sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
        sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=seed, device="cuda")
        del Xtr, ytr
        pte, pva = W.predict(m, best, Xte), W.predict(m, best, Xva)
        if ctrl is None:
            ev, evv = W.evaluate(sub_te, yte, pte), W.evaluate(sub_va, yva, pva)
        else:
            ev, evv = W.evaluate(sub_te, yte, pte, ref=ctrl["pte"]), W.evaluate(sub_va, yva, pva, ref=ctrl["pva"])
        res = {"name": name, "seed": seed, "n_features": len(feats), "best_iteration": int(best),
               "validation_logloss": evv["logloss"], "test_logloss": ev["logloss"]}
        if ctrl is not None:
            res.update({"validation_delta_vs_control": evv["delta_vs_ref"], "test_delta_vs_control": ev["delta_vs_ref"],
                        "test_delta_ci95": ev["delta_vs_ref_ci95"], "validation_delta_ci95": evv["delta_vs_ref_ci95"]})
        return res, {"pte": pte, "pva": pva}

    out = {"seeds": SEEDS, "variants": []}
    for seed in SEEDS:
        ctrl_res, cp = fit_eval("control (59)", feats59, seed)
        out["variants"].append(ctrl_res)
        print(f"seed {seed} control: validation {ctrl_res['validation_logloss']:.6f}; test {ctrl_res['test_logloss']:.6f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        for name, feats in variants[1:]:
            r, _ = fit_eval(name, feats, seed, cp)
            out["variants"].append(r)
            lo, hi = r["test_delta_ci95"]
            print(f"seed {seed} {name}: validation {r['validation_logloss']:.6f} (delta {r['validation_delta_vs_control']:+.5f}); "
                  f"test {r['test_logloss']:.6f} (delta {r['test_delta_vs_control']:+.5f} [{lo:+.5f}, {hi:+.5f}]) ({time.time() - t0:.0f}s)",
                  flush=True)
            W.dump_json(out, os.path.join(d, "exp_scorelead_combo.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
