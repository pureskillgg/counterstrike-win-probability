"""Final models on the CURRENT split manifest (data/combined/split_manifest.json), every second, with the manifest hash recorded.

  ...python.exe -I -X utf8 scripts\\model\\train_final.py --model basic|full [--config cfg.json] [--device cpu] [--threads 32]

basic : winprob_v4_basic, the 13 event inputs of winprob_v3_basic, trained on the grid of every era (2024 + 2026).
full  : winprob_v4, the 59 snapshot inputs of winprob_v3, trained on the 2026 snapshot cache.
Train = the manifest's train matches. Validation = its validation matches (early stopping, and the score used to choose a configuration).
Test = its test matches, scored once. Every second is used; the rounds with a missing player are dropped (wp_common.drop_roster_defect_rounds).
Writes models/<name>.json and models/<name>.meta.json. The meta records the manifest hash and the input-file hash it was trained on.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]
BASIC_FEATS = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live", "seconds_since_plant"] + [f"map_is_{m}" for m in MAPS]
BASELINE = {"depth": 4, "eta": 0.05, "mcw": 20, "subsample": 0.8}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_basic(store):
    cols = ["match_key", "round", "tick", "second", "alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
            "seconds_since_plant", "label_ct_win", "w_round", "era", "map_name"]
    G = W.load_grid(store, cols=cols)
    names = set(G["map_name"].unique().to_list())
    expected = {f"de_{m}" for m in MAPS}
    if not names & expected:
        sys.exit(f"no modelled map names in the grid: {sorted(names)[:10]}")
    G = G.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Float32).alias(f"map_is_{m}") for m in MAPS])
    return W.drop_roster_defect_rounds(G, store), BASIC_FEATS


def load_full(store):
    d = os.path.join(W.ROOT, "data", store)
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    G = pl.read_parquet(os.path.join(d, "cache_snapshot_2026.parquet"))
    return W.drop_roster_defect_rounds(G, store), feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["basic", "full"])
    ap.add_argument("--config", default=None, help="JSON with depth, eta, mcw, subsample and any extra XGBoost parameters")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--store", default="combined")
    ap.add_argument("--seed", type=int, default=0, help="XGBoost seed (seed variance)")
    ap.add_argument("--no-mono", action="store_true", help="ablation: no monotone constraints")
    ap.add_argument("--patience", type=int, default=100, help="early-stopping rounds on validation log loss")
    a = ap.parse_args()
    tag = ("" if a.seed == 0 else f"_s{a.seed}") + ("_nomono" if a.no_mono else "") + ("" if a.patience == 100 else f"_es{a.patience}")
    cfg = dict(BASELINE)
    if a.config:
        cfg.update(json.load(open(a.config, encoding="utf-8")))
    d = os.path.join(W.ROOT, "data", a.store)
    manifest = os.path.join(d, "split_manifest.json")
    t0 = time.time()

    if a.model == "basic":
        name = "winprob_v4_basic"
        G, feats = load_basic(a.store)
        input_file = os.path.join(d, "grid.parquet")
    else:
        name = "winprob_v4"
        G, feats = load_full(a.store)
        input_file = os.path.join(d, "cache_snapshot_2026.parquet")

    sp = W.make_split("random", G, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    mono = [0] * len(feats) if a.no_mono else list(W.mono_for(feats))
    name = name + tag
    _, Xtr, ytr = W.arrays(G, tr, feats, stride=1)
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
    print(f"{name}: train {len(tr)} matches / {len(ytr):,} seconds; validation {len(va)} / {len(yva):,}; test {len(te)} / {len(yte):,} "
          f"({time.time() - t0:.0f}s)", flush=True)

    extra = {k: v for k, v in cfg.items() if k not in ("depth", "eta", "mcw", "subsample")}
    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=a.seed, device=a.device, depth=cfg["depth"], eta=cfg["eta"], mcw=cfg["mcw"],
                        subsample=cfg["subsample"], n_jobs=a.threads, early_stopping_rounds=a.patience, **extra)
    print(f"trained: best iteration {best} ({time.time() - t0:.0f}s)", flush=True)

    pva = W.predict(m, best, Xva)
    pte = W.predict(m, best, Xte)
    ev_va = W.evaluate(sub_va, yva, pva)
    ev_te = W.evaluate(sub_te, yte, pte)
    print(f"validation log loss {ev_va['logloss']:.5f}; test log loss {ev_te['logloss']:.5f} "
          f"{[round(x, 5) for x in ev_te['logloss_ci95']]} AUC {ev_te['auc']:.4f} ECE10 {ev_te['ece10']:.4f}", flush=True)

    os.makedirs(os.path.join(W.ROOT, "models"), exist_ok=True)
    m.get_booster().save_model(os.path.join(W.ROOT, "models", f"{name}.json"))
    meta = {
        "version": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "features": feats,
        "monotone_constraints": dict(zip(feats, mono)), "best_iteration": int(best), "params": cfg, "xgboost": xgb.__version__,
        "seed": a.seed, "no_mono": a.no_mono, "patience": a.patience, "stride": 1, "device": a.device, "split": "random", "split_manifest_sha256": sha256(manifest),
        "input_file": os.path.basename(input_file), "input_file_sha256": sha256(input_file), "drop_roster_defect_rounds": True,
        "training": {"matches": len(tr), "val_matches": len(va), "rows": int(len(ytr))},
        "test": {"matches": len(te), "rows": int(len(yte))},
        "validation_logloss": ev_va["logloss"], "test_logloss": ev_te["logloss"], "test_logloss_ci95": ev_te["logloss_ci95"],
        "test_logloss_round_weighted": ev_te["logloss_round_weighted"], "test_auc": ev_te["auc"], "test_ece10": ev_te["ece10"],
        "test_brier": ev_te["brier"],
    }
    with open(os.path.join(W.ROOT, "models", f"{name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"done ({time.time() - t0:.0f}s): models/{name}.json", flush=True)


if __name__ == "__main__":
    main()
