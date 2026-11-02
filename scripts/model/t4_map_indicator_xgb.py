"""T4: does a one-hot map indicator add to the train-only-centre distance XGBoost (SN), beyond the per-map normalised distances?

Pre-registration (Meridian pin 60db5389, written before any result was read):
  Arms (validation seconds, decision set): SN = 57 frozen inputs plus straight-line distances from train-only site centres
  (data/diagnostics/allmaps_geo_trainsites); SNM = SN plus seven one-hot map indicators. Same rows, same recipe, same early stopping.
  Rule (standard): pass if for EACH seed mean(SNM minus SN) on validation <= -0.0005 with a 95% match-clustered CI upper bound < 0,
  and the three-seed mean <= -0.0005. Test seconds are descriptive only.

Runs on the RunPod pod or locally. The root is read from CS2_ROOT (default: the local project). Needs polars, numpy, xgboost.
Writes data/diagnostics/t4_map_indicator.json and data/diagnostics/preds/t4_snm_seed<s>.parquet.
  python -I -X utf8 -u scripts/model/t4_map_indicator_xgb.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

ROOT = os.environ.get("CS2_ROOT", "G:/pureskill/csds-cs2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp_common as W  # noqa: E402

W.ROOT = ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
PREDS = os.path.join(DIAG, "preds")
OUT = os.path.join(DIAG, "t4_map_indicator.json")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEY = ["match_key", "round", "second"]
ST = [f"nst{i}" for i in range(4)]
MAPCOLS = [f"map_{m}" for m in MAPS]
SEEDS = [0, 1, 2]


def bootstrap_paired(match, y, p_a, p_b, seed=0):
    """Mean of (log loss of a minus log loss of b) over seconds; match-cluster bootstrap, 1,000 resamples."""
    la = W.ll_vec(p_a, y)
    lb = W.ll_vec(p_b, y)
    diff = la - lb
    uniq, inv = np.unique(match, return_inverse=True)
    sums, counts = np.zeros(uniq.size), np.zeros(uniq.size)
    np.add.at(sums, inv, diff)
    np.add.at(counts, inv, 1)
    rng = np.random.default_rng(seed)
    boot = np.empty(1000)
    for i in range(1000):
        pick = rng.integers(0, uniq.size, uniq.size)
        boot[i] = sums[pick].sum() / counts[pick].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "rows": int(diff.size), "matches": int(uniq.size)}


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    cols = sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + feats))
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"), columns=cols)
    C = W.drop_roster_defect_rounds(C)
    C = C.filter(pl.col("label_ct_win").is_not_null()).with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    geo = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo_trainsites", f"{m}.parquet")) for m in MAPS])
    geo = geo.rename({f"st{i}": f"nst{i}" for i in range(4)})
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    G = C.join(geo, on=KEY, how="inner").join(A, on="match_key", how="inner")
    G = G.unique(KEY, keep="first", maintain_order=True).sort(KEY)
    G = G.with_columns([(pl.col("map_name") == m).cast(pl.Int8).alias(f"map_{m}") for m in MAPS])
    print(f"joined rows {G.height:,} ({time.time() - t0:.0f} s)", flush=True)

    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter((pl.col("split") == "test") & (pl.col("map_name") == "de_mirage"))
    y_va = va["label_ct_win"].to_numpy().astype(np.int64)
    y_te = te["label_ct_win"].to_numpy().astype(np.int64)
    match_va = va["match_key"].to_numpy()
    match_te = te["match_key"].to_numpy()
    arms = {"SN": feats + ST, "SNM": feats + ST + MAPCOLS}
    os.makedirs(PREDS, exist_ok=True)

    per_seed = {}
    for seed in SEEDS:
        pv, pt = {}, {}
        for name, cols_ in arms.items():
            m, best = W.fit_xgb(tr.select(cols_).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                                va.select(cols_).to_numpy().astype(np.float32), y_va,
                                mono=W.mono_for(cols_), seed=seed, device="cuda")
            pv[name] = W.predict(m, best, va.select(cols_).to_numpy().astype(np.float32))
            pt[name] = W.predict(m, best, te.select(cols_).to_numpy().astype(np.float32))
            print(f"seed {seed} {name}: validation log loss {float(W.ll_vec(pv[name], y_va).mean()):.6f}, "
                  f"test {float(W.ll_vec(pt[name], y_te).mean()):.6f}, best iteration {best} ({time.time() - t0:.0f} s)", flush=True)
            if name == "SNM" and seed == SEEDS[0]:
                # diagnostic (added on the local rerun): total gain on each map column; XGBoost names the columns f<index> here
                gain = m.get_booster().get_score(importance_type="total_gain")
                print("SNM seed 0 total gain on the map indicators:",
                      {c: round(gain.get(f"f{cols_.index(c)}", 0.0), 1) for c in MAPCOLS}, flush=True)
                dump = "\n".join(m.get_booster().get_dump())
                print("SNM seed 0 split nodes on a map indicator:",
                      {c: dump.count(f"[f{cols_.index(c)}<") for c in MAPCOLS}, flush=True)
        pl.DataFrame({"match_key": match_va, "round": va["round"].to_numpy(), "second": va["second"].to_numpy(),
                      "map_name": va["map_name"].to_numpy(), "split": "val", "label": y_va,
                      "p": pv["SNM"].astype(np.float64)}).write_parquet(os.path.join(PREDS, f"t4_snm_val_seed{seed}.parquet"))
        per_seed[seed] = {"SNM_minus_SN_validation": bootstrap_paired(match_va, y_va, pv["SNM"], pv["SN"]),
                          "SNM_minus_SN_test_descriptive": bootstrap_paired(match_te, y_te, pt["SNM"], pt["SN"])}
        d = per_seed[seed]["SNM_minus_SN_validation"]
        print(f"seed {seed}: validation SNM minus SN {d['diff']:+.6f} CI [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]", flush=True)

    d_va = [per_seed[s]["SNM_minus_SN_validation"]["diff"] for s in SEEDS]
    ci_va = [per_seed[s]["SNM_minus_SN_validation"]["ci95"] for s in SEEDS]
    mean = float(np.mean(d_va))
    passes = bool(mean <= -0.0005 and all(c[1] < 0 for c in ci_va))
    res = {"decision_set": "validation seconds", "per_seed": per_seed, "mean_over_seeds_validation": mean, "passes": passes,
           "rule": "for each seed mean <= -0.0005 with CI upper < 0, and three-seed mean <= -0.0005", "root": ROOT,
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"mean_over_seeds_validation": mean, "passes": passes}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
