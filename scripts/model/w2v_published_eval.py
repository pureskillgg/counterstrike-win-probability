"""Word2vec as the published code uses it (codeAnalysis.py lines 169-310), replicated on de_mirage 2026, with a leakage decomposition.

Published protocol, as the code does it:
  1. Sentences: one per round, a state token every 3 seconds from round start. (The published code also skips seconds before the
     first movement; the embedding file used here does not. See --emb.)
  2. Word2Vec(vector_size=8, window=5, min_count=1) fitted on the sentences of ALL matches (leaky).
  3. Rows: one random second per round (any second of the round, not only the 3-second sample points).
  4. A row is KEPT only if its exact state token is in the vocabulary (the code's `except: continue`). Unknown rows are dropped,
     not zero-filled (w2v_eval2.py zero-filled them; this file does not).
  5. Features: num_t_alive, num_ct_alive, t_equip_value, ct_equip_value, plus 8 embedding dimensions.
  6. Split: random 80/20 by row (train_test_split, no grouping by match).
  7. Model: XGBClassifier(max_depth=8, eval_metric='logloss', objective='binary:logistic'), other settings default, no early stopping.
  8. Metrics: accuracy and F1 at threshold 0.5.

Decomposition (all arms use the same kept rows within a cell):
  split  random  = the published row split (kept rows, 80/20 random by row)
         match   = the frozen split (train matches fit, test matches evaluate; validation matches unused)
  vocab  leaky   = embedding fitted on all matches (published)
         honest  = embedding fitted on train matches only
  arms   B  = the 4 published base features only (the published baseline, on the same kept rows)
         E  = B + 8 embedding dimensions
         N  = B + 8 Gaussian noise columns (control: extra columns carrying no information)
  B_all  = B on ALL sampled rows (unknown rows kept), random split only, to show the effect of the drop rule.

Seeds 0, 1, 2: each seed draws its own random second per round, its own random split, and its own model seed.
Paired differences use spatial_feature_eval.paired (match-cluster bootstrap, 1,000 resamples), on log loss.

Pre-registered thresholds (written before any result was read):
  P1 published gain, random split, leaky vocab:    mean(E - B) <= -0.0005 and every seed's 95% CI upper bound < 0
  P2 row split, honest vocab:                      same rule
  P3 frozen protocol, match split, honest vocab:   same rule
  P4 vocabulary leakage only, match split, leaky:  same rule
  Noise control N - B should not pass in any cell.
  E_minus_N (embedding vs noise, equal column count) is a post-hoc diagnostic added after the first run, not a verdict.
Reading: P1 passes and P3 fails -> the published gain does not survive a clean protocol. P1 and P2 pass and P3 fails -> the row
split leaks matches. P1 passes and P2 fails -> the leaky vocabulary is what produces the gain on the random split.

Inputs: the embedding stem (default w2v_gensim: data/diagnostics/{stem}_honest.parquet and {stem}_leaky.parquet, written by
w2v_gensim_embed.py), data/combined/cache_snapshot_2026.parquet, data/combined/split_manifest.json.
Writes data/diagnostics/w2v_published_eval_{stem}.json.

Run in the cs2gpu environment (polars, xgboost):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/w2v_published_eval.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
BASE = ["alive_ct", "alive_t", "st_equip_ct", "st_equip_t"]  # published: num_ct_alive, num_t_alive, ct_equip_value, t_equip_value
EMB = [f"e{i}" for i in range(8)]
NOISE = [f"n{i}" for i in range(8)]
THRESH = -0.0005
DEPTH = 8  # published setting; --depth overrides for the sensitivity run


def keyed(df):
    return df.with_columns([pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)])


def load_embeddings(stem):
    parts = []
    for vocab in ["honest", "leaky"]:
        d = keyed(pl.read_parquet(os.path.join(DIAG, f"{stem}_{vocab}.parquet")))
        d = d.select(KEY + ["known"] + EMB).rename({"known": f"known_{vocab}", **{c: f"{c}_{vocab}" for c in EMB}})
        parts.append(d.unique(KEY, keep="first", maintain_order=True))
    E = parts[0].join(parts[1], on=KEY, how="inner").sort(KEY)
    assert E.height == parts[0].height == parts[1].height, (E.height, parts[0].height, parts[1].height)
    return E


def load_cache(feats):
    need = KEY + ["label_ct_win", "map_name"] + feats + BASE
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"), columns=sorted(set(need)))
    C = W.drop_roster_defect_rounds(C)
    C = C.filter((pl.col("map_name") == "de_mirage") & pl.col("label_ct_win").is_not_null())
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    C = keyed(C).join(A, on="match_key", how="inner")
    return C.select(KEY + ["split", "label_ct_win"] + BASE).unique(KEY, keep="first", maintain_order=True).sort(KEY)


def one_per_round(df, rng):
    # Row order is fixed by the sort before any random draw, so a seed reproduces the same sample.
    r = rng.random(df.height)
    return (df.sort(KEY).with_columns(pl.Series("_r", r))
              .sort(["match_key", "round", "_r"])
              .unique(["match_key", "round"], keep="first", maintain_order=True)
              .drop("_r"))


def fit_predict(tr, te, cols, seed):
    m = XGBClassifier(max_depth=DEPTH, eval_metric="logloss", objective="binary:logistic", random_state=seed)
    m.fit(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64))
    return m.predict_proba(te.select(cols).to_numpy().astype(np.float32))[:, 1]


def metrics(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    pred = (p >= 0.5).astype(np.int64)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    return {"logloss": ll, "accuracy": float(np.mean(pred == y)), "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def paired_acc_f1(match, y, pa, pb, n_boot=1000, seed=0):
    """Accuracy and F1 differences (first arm minus second), match-cluster bootstrap with 1,000 resamples and seed 0 (same scheme as
    spatial_feature_eval.paired, which covers log loss only)."""
    uniq, inv = np.unique(match, return_inverse=True)
    M = len(uniq)

    def per_match(p):
        pred = (p >= 0.5).astype(np.int64)

        def s(v):
            return np.bincount(inv, weights=v.astype(np.float64), minlength=M)
        return {"n": s(np.ones(len(y))), "corr": s(pred == y), "tp": s((pred == 1) & (y == 1)),
                "fp": s((pred == 1) & (y == 0)), "fn": s((pred == 0) & (y == 1))}

    def stats(S, idx):
        tp, fp, fn = S["tp"][idx].sum(), S["fp"][idx].sum(), S["fn"][idx].sum()
        acc = S["corr"][idx].sum() / S["n"][idx].sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        return acc, f1

    A, B = per_match(pa), per_match(pb)
    acc_a, f1_a = stats(A, np.arange(M))
    acc_b, f1_b = stats(B, np.arange(M))
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, 2))
    for k in range(n_boot):
        idx = rng.integers(0, M, M)
        a1, f1a = stats(A, idx)
        b1, f1b = stats(B, idx)
        boots[k] = (a1 - b1, f1a - f1b)
    return {"acc_diff": float(acc_a - acc_b),
            "acc_ci95": [float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))],
            "f1_diff": float(f1_a - f1_b),
            "f1_ci95": [float(np.percentile(boots[:, 1], 2.5)), float(np.percentile(boots[:, 1], 97.5))]}


def run_cell(tr, te, seed, vocab):
    y = te["label_ct_win"].to_numpy().astype(np.int64)
    match = te["match_key"].to_numpy()
    emb = [f"{c}_{vocab}" for c in EMB]
    cols = {"B": BASE, "E": BASE + emb, "N": BASE + NOISE}
    preds = {k: fit_predict(tr, te, c, seed) for k, c in cols.items()}
    out = {k: metrics(p, y) for k, p in preds.items()}
    out["E_minus_B"] = paired(match, y, preds["E"], preds["B"])
    out["N_minus_B"] = paired(match, y, preds["N"], preds["B"])
    out["E_minus_N"] = paired(match, y, preds["E"], preds["N"])
    out["acc_f1_E_minus_B"] = paired_acc_f1(match, y, preds["E"], preds["B"])
    out["acc_f1_N_minus_B"] = paired_acc_f1(match, y, preds["N"], preds["B"])
    out["n_train"] = tr.height
    out["n_test"] = te.height
    out["test_matches"] = int(te["match_key"].n_unique())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="w2v_gensim")
    ap.add_argument("--out", default=None)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()
    global DEPTH
    DEPTH = args.depth
    tag = "" if DEPTH == 8 else f"_depth{DEPTH}"
    out_path = os.path.join(DIAG, args.out or f"w2v_published_eval_{args.emb}{tag}.json")
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    E = load_embeddings(args.emb)
    C = load_cache(feats)
    base = C.join(E, on=KEY, how="inner").sort(KEY)
    print(f"cache rows {C.height:,}; with embedding rows {base.height:,} ({time.time() - t0:.0f} s)", flush=True)

    per_seed, shares = {}, {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        S = one_per_round(base, rng)
        noise = rng.standard_normal((S.height, 8))
        S = S.with_columns([pl.Series(NOISE[i], noise[:, i]) for i in range(8)])
        shares[seed] = {v: float(S[f"known_{v}"].mean()) for v in ["honest", "leaky"]}
        res = {"sampled_rounds": S.height, "known_share": shares[seed]}

        perm = rng.permutation(S.height)
        cut = int(round(0.8 * S.height))
        Sp = S[perm]
        tr_all, te_all = Sp.slice(0, cut), Sp.slice(cut)
        p_all = fit_predict(tr_all, te_all, BASE, seed)
        res["random_all_rows_baseline"] = {"B_all": metrics(p_all, te_all["label_ct_win"].to_numpy().astype(np.int64)),
                                            "n_train": tr_all.height, "n_test": te_all.height}

        for vocab in ["leaky", "honest"]:
            sub = S.filter(pl.col(f"known_{vocab}") == 1)
            perm2 = rng.permutation(sub.height)
            cut2 = int(round(0.8 * sub.height))
            sp = sub[perm2]
            res[f"random_{vocab}"] = run_cell(sp.slice(0, cut2), sp.slice(cut2), seed, vocab)
            res[f"match_{vocab}"] = run_cell(sub.filter(pl.col("split") == "train"), sub.filter(pl.col("split") == "test"), seed, vocab)
        per_seed[seed] = res
        print(f"seed {seed}: sampled {S.height:,}; known honest {shares[seed]['honest']:.3f}, leaky {shares[seed]['leaky']:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        for cell in ["random_leaky", "random_honest", "match_honest", "match_leaky"]:
            r = res[cell]
            print(f"  {cell:14s} E-B {r['E_minus_B']['diff']:+.6f}  N-B {r['N_minus_B']['diff']:+.6f}  "
                  f"acc B {r['B']['accuracy']:.4f} E {r['E']['accuracy']:.4f}  n_test {r['n_test']:,}", flush=True)

    def rule(cell, key="E_minus_B"):
        d = [per_seed[s][cell][key]["diff"] for s in SEEDS]
        ci = [per_seed[s][cell][key]["ci95"] for s in SEEDS]
        mean = float(np.mean(d))
        return {"mean_over_seeds": mean, "per_seed": d, "per_seed_ci95": ci,
                "passes": bool(mean <= THRESH and all(c[1] < 0 for c in ci))}

    def mean_metric(cell, arm, name):
        return float(np.mean([per_seed[s][cell][arm][name] for s in SEEDS]))

    cells = ["random_leaky", "random_honest", "match_honest", "match_leaky"]
    summary = {c: {arm: {m: mean_metric(c, arm, m) for m in ["accuracy", "f1", "logloss"]} for arm in ["B", "E", "N"]} for c in cells}
    verdicts = {"P1_published_gain_random_leaky": rule("random_leaky"),
                "P2_row_split_honest": rule("random_honest"),
                "P3_frozen_protocol_match_honest": rule("match_honest"),
                "P4_vocab_leak_match_leaky": rule("match_leaky"),
                "noise_control_random_leaky": rule("random_leaky", "N_minus_B"),
                "noise_control_match_honest": rule("match_honest", "N_minus_B")}
    # Post hoc (added after the first run, not pre-registered): embedding vs noise at equal column count, to separate information
    # from the extra capacity that 8 more columns give a depth-8 model.
    diagnostics = {f"E_minus_N_{c}": rule(c, "E_minus_N") for c in cells}
    accuracy_f1 = {c: {"acc_diff_mean": float(np.mean([per_seed[s][c]["acc_f1_E_minus_B"]["acc_diff"] for s in SEEDS])),
                       "f1_diff_mean": float(np.mean([per_seed[s][c]["acc_f1_E_minus_B"]["f1_diff"] for s in SEEDS])),
                       "per_seed_acc_diff": [per_seed[s][c]["acc_f1_E_minus_B"]["acc_diff"] for s in SEEDS],
                       "per_seed_acc_ci95": [per_seed[s][c]["acc_f1_E_minus_B"]["acc_ci95"] for s in SEEDS],
                       "per_seed_f1_diff": [per_seed[s][c]["acc_f1_E_minus_B"]["f1_diff"] for s in SEEDS],
                       "per_seed_f1_ci95": [per_seed[s][c]["acc_f1_E_minus_B"]["f1_ci95"] for s in SEEDS]} for c in cells}
    res = {"emb": args.emb, "depth": DEPTH, "seeds": SEEDS, "known_share": shares, "per_seed": per_seed,
           "summary_mean_over_seeds": summary, "accuracy_f1_E_minus_B": accuracy_f1,
           "verdicts": verdicts, "diagnostics_post_hoc": diagnostics,
           "thresholds": "mean(diff) <= -0.0005 and every seed's CI upper bound < 0",
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: {"mean": v["mean_over_seeds"], "passes": v["passes"]} for k, v in {**verdicts, **diagnostics}.items()},
                     indent=1), flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
