"""Robustness and sufficiency experiments for the snapshot model (GPU, on the cached 2026 feature table).

  stage variants : map handling. A = with the 7 map one-hots (as exported), B = no map input at all, C = map one-hots with 15% of the
                   training MATCHES shown with the map blanked (an "unknown map" learned as a state). Each is scored on the held-out
                   matches with the true map and with the map blanked (= what the server sees for a map it was not trained on).
  stage lomo     : leave one map out. Train without any match of map m, test on the held-out matches of map m, and compare with
                   the reference model A that did see map m (same rows, paired match-cluster interval). This is the honest test of
                   "a map the model has never seen" (a new or non-active-duty map).
  stage curve    : learning curve. Train on 5, 10, 20, 40, 70, 100% of the training matches (nested), fixed test set; log loss and a
                   fitted a + b/sqrt(n) curve to estimate what doubling the data would still buy.
  stage null     : label-shuffle null. Round labels shuffled among training rounds, test on the true labels: log loss must sit at the
                   base rate and AUC near 0.5, otherwise some feature carries label information that is not real signal.

  ...cs2gpu python.exe -I -X utf8 scripts\\model\\exp_robustness.py [--stages variants,lomo,curve,null] [--stride 3] [--curve-variant B]
Results are written after every stage to data/<store>/exp_robustness.json (so a crash keeps the finished stages).
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import json
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]
MAPC = [f"map_is_{m}" for m in MAPS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--stages", default="variants,lomo,curve,null")
    ap.add_argument("--stride", type=int, default=3, help="train on every stride-th second (evaluation always uses all rows)")
    ap.add_argument("--curve-variant", default="B")
    ap.add_argument("--dropout", type=float, default=0.15)
    args = ap.parse_args()
    out_path = os.path.join(W.ROOT, "data", args.store, "exp_robustness.json")
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    G = pl.read_parquet(os.path.join(W.ROOT, "data", args.store, "cache_snapshot_2026.parquet"))
    sp = W.make_split("random", G, seed=0)
    ST = [c for c in G.columns if c.startswith("st_")]
    FEATS = {"A": W.F6 + MAPC + ST, "B": W.F6 + ST, "C": W.F6 + MAPC + ST}
    map_of = dict(G.group_by("match_key").agg(pl.col("map_name").first()).iter_rows())
    print(f"{G.height:,} rows, {len(map_of):,} matches; train/val/test matches {len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}", flush=True)

    def blank_map_for_matches(sub, X, feats, frac, seed):
        """Zero the map one-hots for a random share of MATCHES (whole matches: a map is unknown for a whole match)."""
        cols = [feats.index(c) for c in MAPC]
        mk = sub["match_key"].to_numpy()
        uniq = np.unique(mk)
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(uniq, size=int(round(frac * len(uniq))), replace=False).tolist())
        rows = np.fromiter((k in chosen for k in mk), dtype=bool, count=len(mk))
        X[np.ix_(rows, cols)] = 0.0
        return X

    def fit(variant, train_keys, val_keys, seed=0, y_override=None, stride=None):
        feats = FEATS[variant]
        stride = args.stride if stride is None else stride
        sub_tr, Xtr, ytr = W.arrays(G, train_keys, feats, stride=stride)
        sub_va, Xva, yva = W.arrays(G, val_keys, feats, stride=stride)
        if variant == "C":
            Xtr = blank_map_for_matches(sub_tr, Xtr, feats, args.dropout, seed + 11)
        if y_override is not None:
            # the null model must see no real label anywhere: early stopping on TRUE validation labels lets monotone constraints pick up a little signal
            ytr = y_override(sub_tr)
            yva = y_override(sub_va)
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=seed, device="cuda")
        return m, best, feats, int(len(ytr))

    te_sub, _, yte = W.arrays(G, sp["test"], W.F6)
    models = {}

    def pred(variant, X_from_feats=None, blank_map=False):
        m, best, feats, _ = models[variant]
        X = te_sub.select(feats).to_numpy().astype(np.float32)
        if blank_map and all(c in feats for c in MAPC):
            X[:, [feats.index(c) for c in MAPC]] = 0.0
        return W.predict(m, best, X)

    def summ(r):
        return {"logloss": r["logloss"], "ci95": r["logloss_ci95"], "auc": r["auc"], "ece10": r["ece10"]}

    stages = args.stages.split(",")
    if "variants" in stages:
        t0 = time.time()
        out = {}
        for v in "ABC":
            models[v] = fit(v, sp["train"], sp["val"])
            print(f"variant {v} fitted, best iteration {models[v][1]} ({time.time() - t0:.0f}s)", flush=True)
        pA = pred("A")
        out["test_matches"] = len(sp["test"])
        for v in "ABC":
            p = pred(v)
            r = W.evaluate(te_sub, yte, p, ref=pA if v != "A" else None)
            out[f"{v}_true_map"] = {**summ(r), **({"delta_vs_A": r["delta_vs_ref"], "delta_vs_A_ci95": r["delta_vs_ref_ci95"]} if v != "A" else {})}
            if v != "B":
                pz = pred(v, blank_map=True)
                rz = W.evaluate(te_sub, yte, pz, ref=p)
                out[f"{v}_map_blanked"] = {**summ(rz), "delta_vs_own_true_map": rz["delta_vs_ref"], "delta_ci95": rz["delta_vs_ref_ci95"],
                                           "mean_abs_prob_shift": float(np.mean(np.abs(pz - p))), "p99_abs_prob_shift": float(np.percentile(np.abs(pz - p), 99)),
                                           "max_abs_prob_shift": float(np.max(np.abs(pz - p)))}
        res["variants"] = out
        W.dump_json(res, out_path)
        print(json.dumps({k: (v["logloss"] if isinstance(v, dict) and "logloss" in v else None) for k, v in out.items() if isinstance(v, dict)}, indent=1), flush=True)

    if "lomo" in stages:
        if "A" not in models:
            models["A"] = fit("A", sp["train"], sp["val"])
        pA = pred("A")
        out = dict(res.get("lomo", {}))  # maps finished by an earlier run are kept and skipped
        for m_name in MAPS:
            if m_name in out:
                print(f"lomo {m_name}: already done", flush=True)
                continue
            t0 = time.time()
            tr = {k for k in sp["train"] if map_of[k] != "de_" + m_name}
            va = {k for k in sp["val"] if map_of[k] != "de_" + m_name}
            mask = (te_sub["map_name"] == "de_" + m_name).to_numpy()
            sub_m = te_sub.filter(pl.Series(mask))
            y_m = yte[mask]
            row = {"train_matches": len(tr), "test_rows": int(mask.sum()), "test_matches": int(sub_m["match_key"].n_unique()),
                   "reference_A_saw_map": summ(W.evaluate(sub_m, y_m, pA[mask]))}
            for v in "ABC":
                mm, best, feats, n = fit(v, tr, va)
                X = sub_m.select(feats).to_numpy().astype(np.float32)
                p = W.predict(mm, best, X)
                r = W.evaluate(sub_m, y_m, p, ref=pA[mask])
                row[f"{v}_never_saw_map"] = {**summ(r), "cost_vs_reference": r["delta_vs_ref"], "cost_ci95": r["delta_vs_ref_ci95"]}
            out[m_name] = row
            print(f"lomo {m_name}: " + "  ".join(f"{v} {row[f'{v}_never_saw_map']['logloss']:.4f} ({row[f'{v}_never_saw_map']['cost_vs_reference']:+.4f})" for v in "ABC") +
                  f"  ref {row['reference_A_saw_map']['logloss']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            res["lomo"] = out
            W.dump_json(res, out_path)

    if "curve" in stages:
        v = args.curve_variant
        order = sorted(sp["train"])
        np.random.default_rng(5).shuffle(order)
        pts = []
        for frac in (0.05, 0.10, 0.20, 0.40, 0.70, 1.00):
            t0 = time.time()
            keys = set(order[:max(2, int(round(frac * len(order))))])
            mm, best, feats, n_rows = fit(v, keys, sp["val"])
            p = W.predict(mm, best, te_sub.select(feats).to_numpy().astype(np.float32))
            r = W.evaluate(te_sub, yte, p)
            pts.append({"fraction": frac, "train_matches": len(keys), "train_rows_used": n_rows, **summ(r)})
            print(f"curve {int(frac * 100):>3}%: {len(keys)} matches  LL {r['logloss']:.4f} {r['logloss_ci95']}  ({time.time() - t0:.0f}s)", flush=True)
        n = np.array([q["train_matches"] for q in pts], dtype=float)
        y = np.array([q["logloss"] for q in pts])
        A = np.stack([np.ones_like(n), 1.0 / np.sqrt(n)], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        fit_at = lambda nn: float(coef[0] + coef[1] / np.sqrt(nn))
        res["curve"] = {"variant": v, "points": pts, "fit_a_plus_b_over_sqrt_n": {"a": float(coef[0]), "b": float(coef[1])},
                        "predicted_logloss_at_current_n": fit_at(n[-1]), "predicted_gain_from_doubling": fit_at(n[-1]) - fit_at(2 * n[-1]),
                        "predicted_gain_from_adding_1000_matches": fit_at(n[-1]) - fit_at(n[-1] + 1000), "asymptote": float(coef[0])}
        W.dump_json(res, out_path)
        print("curve fit:", {k: res["curve"][k] for k in ("predicted_gain_from_doubling", "predicted_gain_from_adding_1000_matches", "asymptote")}, flush=True)

    if "null" in stages:
        rng = np.random.default_rng(9)

        def shuffled(sub):
            rk = sub.select("match_key", "round", "label_ct_win").unique(["match_key", "round"])
            perm = rng.permutation(rk["label_ct_win"].to_numpy())
            lab = rk.with_columns(pl.Series("y_shuf", perm)).select("match_key", "round", "y_shuf")
            return sub.join(lab, on=["match_key", "round"], how="left")["y_shuf"].to_numpy().astype(np.int64)

        mm, best, feats, _ = fit("B", sp["train"], sp["val"], y_override=shuffled)
        p = W.predict(mm, best, te_sub.select(feats).to_numpy().astype(np.float32))
        r = W.evaluate(te_sub, yte, p)
        base = float(yte.mean())
        const = float(-np.mean(yte * np.log(base) + (1 - yte) * np.log(1 - base)))
        res["null"] = {"test_logloss_shuffled_label_model": r["logloss"], "ci95": r["logloss_ci95"], "auc": r["auc"], "constant_logloss_on_test": const,
                       "best_iteration": best}
        W.dump_json(res, out_path)
        print(f"null: LL {r['logloss']:.4f} (constant {const:.4f})  AUC {r['auc']:.3f}", flush=True)
    print("-> ", out_path)


if __name__ == "__main__":
    main()
