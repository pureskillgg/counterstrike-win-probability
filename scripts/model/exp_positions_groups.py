"""Do position features help more for some round types? Both models in one run, on the same seconds.   (XGBoost, GPU)

  ...python.exe -I -X utf8 scripts\\model\\exp_positions_groups.py [--store combined] [--device cuda]

base59 : the 59 snapshot inputs
pos83  : base59 + 24 position / view / blindness inputs (joined table, built from cache_pos_2026.parquet with --every 1)
Both are trained on every second of the training matches with the same settings, and early-stopped on the validation matches.
Predictions are made block by block, so the 83-column matrices for validation and test are never held in memory at once.
Each round group is scored separately, on validation and on test: pistol (rounds 1, 13), after pistol (2, 14), rest (3-12, 15-24, overtime).
The pos83 minus base59 difference per group is paired on the same seconds, with a match-cluster 95% interval.
Checkpoints: once a model is trained, its validation and test predictions are saved to data/<store>/ckpt/ with a fingerprint of its
settings (features, monotone constraints, XGBoost parameters, the split). A rerun loads a checkpoint only when the fingerprint matches;
otherwise that model is trained again. A job that dies after the base model keeps that model.
Writes data/<store>/exp_positions_groups.json.
"""
import argparse
import json
import os
import sys
import time

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import wp_common as W
from exp_full_positions import POS, joined_file
from exp_pistol import BASE, train

sys.stdout.reconfigure(encoding="utf-8")
GROUPS = {"pistol": ([1, 13], None), "after_pistol": ([2, 14], None), "rest": (None, [1, 13, 2, 14])}


def predict_blocks(bst, best, lf, keys, cols, chunk=300):
    """Predictions, labels, rounds, match keys and weights for some matches, one block at a time."""
    ps, ys, rs, mks, ws = [], [], [], [], []
    for c0 in range(0, len(keys), chunk):
        ck = keys[c0:c0 + chunk]
        df = lf.filter(pl.col("match_key").is_in(ck)).select(["match_key", "round", "label_ct_win", "w_round"] + [pl.col(c).cast(pl.Float32) for c in cols]).collect()
        X = df.select(cols).to_numpy().astype(np.float32)
        ps.append(bst.predict(xgb.DMatrix(X), iteration_range=(0, best + 1)))
        ys.append(df["label_ct_win"].to_numpy())
        rs.append(df["round"].to_numpy())
        mks.append(df["match_key"].to_numpy())
        ws.append(df["w_round"].to_numpy())
        del df, X
    return np.concatenate(ps), np.concatenate(ys).astype(np.int64), np.concatenate(rs).astype(np.int64), np.concatenate(mks).astype(str), np.concatenate(ws)


def save_ckpt(d, name, fp, **arrays):
    os.makedirs(os.path.join(d, "ckpt"), exist_ok=True)
    tmp = os.path.join(d, "ckpt", f"{name}.tmp.npz")
    np.savez(tmp, fp=np.array(fp), **arrays)
    os.replace(tmp, os.path.join(d, "ckpt", f"{name}.npz"))


def load_ckpt(d, name, fp):
    """The saved arrays when the checkpoint was made with the same settings; None otherwise."""
    path = os.path.join(d, "ckpt", f"{name}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    if str(z["fp"]) != fp:
        print(f"{name}: checkpoint was made with other settings or split: training again", flush=True)
        return None
    print(f"{name}: loaded from checkpoint", flush=True)
    return {k: z[k] for k in z.files if k != "fp"}


def stage_base(d, name, fp, lf, tr, va, te, feats, mono, device):
    cached = load_ckpt(d, name, fp)
    if cached is not None:
        return cached
    bst, best, s = train(lf, tr, va, feats, False, mono, None, False, {"seed": 0}, device)
    pv, yv, rv, mv, wv = predict_blocks(bst, best, lf, va, feats)
    pt, yt, rt, mt, wt = predict_blocks(bst, best, lf, te, feats)
    out = {"pv": pv, "yv": yv, "rv": rv, "mv": mv, "wv": wv, "pt": pt, "yt": yt, "rt": rt, "mt": mt, "wt": wt,
           "best": np.array(best), "secs": np.array(s)}
    save_ckpt(d, name, fp, **out)
    print(f"{name} trained: best {best} ({s}s)", flush=True)
    return out


def stage_pos(d, name, fp, lf, tr, va, te, cols, mono, device):
    cached = load_ckpt(d, name, fp)
    if cached is not None:
        return cached
    bst, best, s = train(lf, tr, va, cols, False, mono, None, False, {"seed": 0}, device)
    pv, _, _, _, _ = predict_blocks(bst, best, lf, va, cols)
    pt, _, _, _, _ = predict_blocks(bst, best, lf, te, cols)
    out = {"pv": pv, "pt": pt, "best": np.array(best), "secs": np.array(s)}
    save_ckpt(d, name, fp, **out)
    print(f"{name} trained: best {best} ({s}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    t0 = time.time()
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    cols83 = feats + POS
    joined = joined_file(os.path.join(d, "cache_snapshot_2026.parquet"), os.path.join(d, "cache_pos_2026.parquet"), feats,
                         os.path.join(d, "_joined_pos_2026.parquet"))
    lf = pl.scan_parquet(joined)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    del kdf
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    mono = list(W.mono_for(feats))
    mono_pos = mono + [0] * len(POS)
    fp_base = W.fingerprint("base59", feats, mono, BASE, tr, va, te)
    fp_pos = W.fingerprint("pos83", cols83, mono_pos, BASE, tr, va, te)
    print(f"joined ready ({time.time() - t0:.0f}s); val {len(va)} test {len(te)} matches", flush=True)

    B = stage_base(d, "base59", fp_base, lf, tr, va, te, feats, mono, a.device)
    P = stage_pos(d, "pos83", fp_pos, lf, tr, va, te, cols83, mono_pos, a.device)
    print(f"base59 best {int(B['best'])}; pos83 best {int(P['best'])} ({time.time() - t0:.0f}s)", flush=True)
    yva, rva, mkva, wva, pb_va, pp_va = B["yv"], B["rv"], B["mv"], B["wv"], B["pv"], P["pv"]
    yte, rte, mkte, wte, pb_te, pp_te = B["yt"], B["rt"], B["mt"], B["wt"], B["pt"], P["pt"]

    def mask_for(r, spec):
        inc, exc = spec
        if inc is not None:
            return np.isin(r, inc)
        return ~np.isin(r, exc)

    out = {"base_best_iteration": int(B["best"]), "pos_best_iteration": int(P["best"]), "groups": {}}
    for split_name, mk, w, y, pb, pp, r in (("validation", mkva, wva, yva, pb_va, pp_va, rva), ("test", mkte, wte, yte, pb_te, pp_te, rte)):
        sub = pl.DataFrame({"match_key": mk, "w_round": w})
        out["groups"][split_name] = {}
        ov = W.evaluate(sub, y, pp, ref=pb)
        out["groups"][split_name]["overall"] = {"seconds": int(len(y)), "base59": W.evaluate(sub, y, pb)["logloss"], "pos83": ov["logloss"],
                                                 "delta": ov["delta_vs_ref"], "delta_ci95": ov["delta_vs_ref_ci95"]}
        for g, spec in GROUPS.items():
            m = mask_for(r, spec)
            sg = sub.filter(pl.Series(m))
            rg = W.evaluate(sg, y[m], pp[m], ref=pb[m])
            out["groups"][split_name][g] = {"seconds": int(m.sum()), "base59": W.evaluate(sg, y[m], pb[m])["logloss"], "pos83": rg["logloss"],
                                            "delta": rg["delta_vs_ref"], "delta_ci95": rg["delta_vs_ref_ci95"]}
        print(f"{split_name}:", flush=True)
        for g in ("overall", *GROUPS):
            e = out["groups"][split_name][g]
            print(f"  {g:13s} base59 {e['base59']:.5f}  pos83 {e['pos83']:.5f}  delta {e['delta']:+.5f} [{e['delta_ci95'][0]:+.5f}, {e['delta_ci95'][1]:+.5f}]",
                  flush=True)
    W.dump_json(out, os.path.join(d, "exp_positions_groups.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
