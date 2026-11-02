"""Do player positions and view angles help? (GPU, paired, on the frozen split)

  ...cs2gpu python.exe -I -X utf8 scripts\\model\\exp_positions.py [--store combined]

Three models with identical settings, trained on the same frozen-split training matches and scored on the same frozen-split TEST matches, on the seconds where the
position table exists (every other second, see scripts/data/make_pos_features.py):
  base      : the 59 snapshot inputs of the final model (alive counts, clock, bomb, map, loadout totals)
  geometry  : base + site occupancy, spacing, contact distance and the bomb / defuse features   (no view angles)
  geometry+view : geometry + how many players have an enemy in their view cone and how many are 'exposed' (an enemy that sees them while they do not see it)
  geometry+view+blind : + how many players are blinded by a flash and for how long
The paired log-loss difference to base carries a match-cluster bootstrap interval. Writes data/<store>/exp_positions.json.
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
GEO = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_nearest_site_mean", "t_nearest_site_mean", "c_spread", "t_spread", "ct_t_centroid_dist", "min_enemy_dist",
       "c_in_contact", "t_in_contact", "c_min_bomb_dist", "c_n_bomb_500", "t_min_bomb_dist", "defuse_slack"]
VIEW = ["c_sees_enemy", "t_sees_enemy", "c_exposed", "t_exposed"]
BLIND = ["c_blind_n", "t_blind_n", "c_blind_sum", "t_blind_sum"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--max-train-rows", type=int, default=0)
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    G = pl.read_parquet(os.path.join(d, "cache_snapshot_2026.parquet"))
    P = pl.read_parquet(os.path.join(d, "cache_pos_2026.parquet"))
    G = G.join(P, on=["match_key", "round", "tick"], how="inner")
    sp = W.make_split("random", G, seed=0)
    ST = [c for c in G.columns if c.startswith("st_")]
    base = W.F6 + [c for c in G.columns if c.startswith("map_is_")] + ST  # the 59 inputs of the final model
    sets = {"base": base, "geometry": base + GEO, "geometry+view": base + GEO + VIEW, "geometry+view+blind": base + GEO + VIEW + BLIND}
    print(f"rows with positions {G.height:,}; matches {G['match_key'].n_unique()}; train {len(sp['train'])} val {len(sp['val'])} test {len(sp['test'])}", flush=True)
    models = {}
    for name, feats in sets.items():
        t0 = time.time()
        _, Xtr, ytr = W.arrays(G, sp["train"], feats, stride=1)
        _, Xva, yva = W.arrays(G, sp["val"], feats, stride=1)
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=0, device="cuda")
        models[name] = (m, best, feats)
        print(f"{name}: {len(feats)} inputs, {len(ytr):,} training seconds, best iteration {best} ({time.time() - t0:.0f}s)", flush=True)
    keep = ("logloss", "logloss_ci95", "auc", "ece10", "ece10_ci95", "smooth_ece")
    out = {"rows_with_positions": G.height, "train_matches": len(sp["train"]), "test_matches": len(sp["test"]), "inputs": {k: len(v) for k, v in sets.items()}}
    ref_p, sub0, y0 = None, None, None
    for name, (m, best, feats) in models.items():
        sub, X, y = W.arrays(G, sp["test"], feats, stride=1)
        p = W.predict(m, best, X)
        if ref_p is None:
            ref_p, sub0 = p, sub
            r = W.evaluate(sub, y, p)
            out[name] = {k: r[k] for k in keep}
            out["test_seconds"] = r["n_rows"]
        else:
            r = W.evaluate(sub, y, p, ref=ref_p)
            out[name] = {k: r[k] for k in keep}
            out[name]["delta_vs_base"] = r["delta_vs_ref"]
            out[name]["delta_vs_base_ci95"] = r["delta_vs_ref_ci95"]
        print(f"{name}: test log loss {out[name]['logloss']:.4f}" + (f"; delta vs base {out[name]['delta_vs_base']:+.4f} {[round(x, 4) for x in out[name]['delta_vs_base_ci95']]}" if "delta_vs_base" in out[name] else ""), flush=True)
    # importance of the added inputs in the fullest model (gain)
    m, best, feats = models["geometry+view+blind"]
    try:
        gain = m.get_booster().get_score(importance_type="total_gain")
        tot = sum(gain.values()) or 1.0
        out["gain_share"] = {f: round(gain.get(f"f{i}", gain.get(f, 0.0)) / tot, 4) for i, f in enumerate(feats) if f in GEO + VIEW + BLIND}
    except Exception as exc:  # noqa: BLE001
        out["gain_share_error"] = repr(exc)[:200]
    W.dump_json(out, os.path.join(d, "exp_positions.json"))


if __name__ == "__main__":
    main()
